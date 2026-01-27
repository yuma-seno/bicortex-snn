import numpy as np

class CLIColors:
    RESET = "\033[0m"
    GRAY = "\033[90m"
    BLUE = "\033[94m"    # Low activity
    GREEN = "\033[92m"   # Medium
    YELLOW = "\033[93m"  # High
    RED = "\033[91m"     # Very High
    BOLD = "\033[1m"
    CYAN = "\033[96m"

def print_cli_heatmap(spike_history: np.ndarray, bin_size: int = 10, title: str = "Activity Log"):
    """スパイク履歴(0/1)のヒートマップ表示"""
    if spike_history.ndim > 1:
        data = np.sum(spike_history, axis=1)
    else:
        data = spike_history

    total_steps = len(data)
    n_bins = int(np.ceil(total_steps / bin_size))
    
    print(f"\n{CLIColors.BOLD}=== {title} (Bin: {bin_size}) ==={CLIColors.RESET}")
    output_buffer = ""
    
    for i in range(n_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, total_steps)
        count = int(np.sum(data[start:end]))
        
        char = str(count) if count < 10 else "+"
        if count == 0: color = CLIColors.GRAY; char = "_"
        elif count <= 2: color = CLIColors.BLUE
        elif count <= 5: color = CLIColors.GREEN
        elif count <= 9: color = CLIColors.YELLOW
        else: color = CLIColors.RED + CLIColors.BOLD
            
        output_buffer += f"{color}{char}{CLIColors.RESET}"
        if (i + 1) % 100 == 0: output_buffer += "\n"

    print(output_buffer)

def print_cli_float_series(data_series: np.ndarray, bin_size: int = 10, title: str = "Value Log"):
    """
    連続値（重みなど）の推移を簡易グラフ化して表示する
    値の大きさに応じて高さを表現する文字 (  ▂ ▃ ▄ ▅ ▆ ▇ █ ) を使用
    """
    # データをビンごとに平均化
    total_steps = len(data_series)
    n_bins = int(np.ceil(total_steps / bin_size))
    binned_data = []
    
    for i in range(n_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, total_steps)
        binned_data.append(np.mean(data_series[start:end]))
    
    binned_data = np.array(binned_data)
    min_val = np.min(binned_data)
    max_val = np.max(binned_data)
    range_val = max_val - min_val if max_val > min_val else 1.0
    
    # Unicode Block Elements
    blocks = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    
    print(f"\n{CLIColors.BOLD}=== {title} (Min: {min_val:.4f} -> Max: {max_val:.4f}) ==={CLIColors.RESET}")
    output_buffer = ""
    
    for val in binned_data:
        # 正規化 0.0 - 7.99
        norm_idx = int(((val - min_val) / range_val) * 7.99)
        char = blocks[norm_idx]
        
        # 色分け (後半ほど赤くする＝学習が進んでいる感)
        if norm_idx < 2: color = CLIColors.BLUE
        elif norm_idx < 5: color = CLIColors.GREEN
        else: color = CLIColors.RED
        
        output_buffer += f"{color}{char}{CLIColors.RESET}"
        
        if len(output_buffer.replace(CLIColors.BLUE, "").replace(CLIColors.RESET, "")) % 100 == 0: # 簡易改行判定
           pass # エスケープシーケンスがあるため正確な改行は難しいが、今回はそのまま流す

    # 100文字ごとに改行を入れる処理（エスケープシーケンス除去は複雑なので簡易的に）
    # 今回は改行なしで一行または端末折り返しに任せる
    print(output_buffer)