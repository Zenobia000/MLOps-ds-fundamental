"""LSTM 產能需求預測模型（PyTorch）。

場景：以過去 ``lookback`` 步的單變量需求序列，預測未來 ``horizon`` 步。
提供 fit / predict / save / load 一致介面，並含「無 GPU 後援」（自動偵測
``torch.cuda`` 可用性，無 GPU 時退回 CPU）。小樣本可在 CPU 上數秒內跑完。

TODO（生產化）：
    - 多變量輸入、外生變數（exogenous）、機台分群（per-entity）建模。
    - 標準化 / 反標準化 pipeline 與標準化參數一併序列化。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.logging import get_logger

logger = get_logger(__name__)

WEIGHTS_FILENAME = "lstm.pt"


def make_windows(
    series: np.ndarray, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """把一維序列切成監督式樣本 ``(X, y)``。

    Args:
        series:   一維數值序列。
        lookback: 回看視窗長度（輸入步數）。
        horizon:  預測步數（輸出步數）。

    Returns:
        ``X`` 形狀 ``(n, lookback, 1)``、``y`` 形狀 ``(n, horizon)``。
    """
    values = np.asarray(series, dtype=np.float32).reshape(-1)
    n = len(values) - lookback - horizon + 1
    if n <= 0:
        raise ValueError(
            f"序列長度 {len(values)} 不足以切出視窗（lookback={lookback}, horizon={horizon}）。"
        )
    xs = np.stack([values[i : i + lookback] for i in range(n)])
    ys = np.stack([values[i + lookback : i + lookback + horizon] for i in range(n)])
    return xs[..., np.newaxis], ys


class _LSTMNet(nn.Module):
    """內部 LSTM 迴歸網路：取最後時間步隱狀態接全連接輸出 horizon。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])  # 取最後時間步


class LSTMForecaster:
    """LSTM 需求預測封裝，介面對齊其餘模型（fit / predict / save / load）。"""

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        p = dict(params or {})
        self.input_size = int(p.get("input_size", 1))
        self.hidden_size = int(p.get("hidden_size", 32))
        self.num_layers = int(p.get("num_layers", 1))
        self.output_size = int(p.get("output_size", p.get("horizon", 1)))
        self.dropout = float(p.get("dropout", 0.0))
        self.epochs = int(p.get("epochs", 20))
        self.lr = float(p.get("lr", p.get("learning_rate", 0.01)))
        self.batch_size = int(p.get("batch_size", 16))

        # 無 GPU 後援：偵測 CUDA，否則 CPU。
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = _LSTMNet(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.output_size,
            self.dropout,
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMForecaster":
        """訓練模型（MSE loss + Adam）。回傳 self。"""
        ds = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        optim = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self._net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim.zero_grad()
                loss = loss_fn(self._net(xb), yb)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * len(xb)
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                logger.info("LSTM epoch %d/%d loss=%.5f", epoch + 1, self.epochs, epoch_loss / len(ds))
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測未來 horizon 步，回傳 ``(n, horizon)`` 陣列。"""
        self._net.eval()
        tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._net(tensor).cpu().numpy()

    def save(self, dir_path: str | Path) -> Path:
        """序列化網路權重與結構超參到 ``<dir_path>/lstm.pt``。"""
        out_dir = Path(dir_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        weights = out_dir / WEIGHTS_FILENAME
        config = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout": self.dropout,
        }
        torch.save({"state_dict": self._net.state_dict(), "config": config}, weights)
        logger.info("已儲存 LSTM 模型：%s", weights)
        return weights

    @classmethod
    def load(cls, dir_path: str | Path) -> "LSTMForecaster":
        """從 ``<dir_path>/lstm.pt`` 載入模型（CPU 安全）。"""
        weights = Path(dir_path) / WEIGHTS_FILENAME
        if not weights.exists():
            raise FileNotFoundError(f"找不到模型權重：{weights}")
        ckpt = torch.load(weights, map_location="cpu")
        instance = cls(params=ckpt["config"])
        instance._net.load_state_dict(ckpt["state_dict"])
        logger.info("已載入 LSTM 模型：%s", weights)
        return instance
