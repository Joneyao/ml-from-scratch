"""matplotlib 中文字体与通用绘图辅助（仓库自包含，零外部依赖）。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_CJK_CANDIDATES = [
    "Noto Sans CJK JP", "Noto Sans CJK SC", "Hiragino Sans GB",
    "PingFang SC", "STHeiti", "Microsoft YaHei", "Arial Unicode MS",
]


def setup_cjk_font() -> str:
    """按优先级选一个可用的中文字体，返回字体名。找不到则退回默认并只关掉负号方框。"""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    fam = plt.rcParams["font.family"]
    return fam[0] if isinstance(fam, (list, tuple)) else str(fam)


def save_fig(fig, out_dir: str, filename: str) -> str:
    """保存图到 out_dir/filename，返回完整路径。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
