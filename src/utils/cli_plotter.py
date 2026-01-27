import numpy as np

class CLIColors:
    RESET = "\033[0m"
    GRAY = "\033[90m"
    BLUE = "\033[94m"    # Low activity
    GREEN = "\033[92m"   # Medium
    YELLOW = "\033[93m"  # High
    RED = "\033[91m"     # Very High
    BOLD = "\033[1m"

def print_cli_heatmap(spike_history: np.ndarray, bin_size: int = 10, title: str = "Activity Log"):
    """
    スパイク履歴をビンごとに集計し、数値と色でコンソールに表示する。
    
    Args:
        spike_history (np.ndarray): (Steps, Neurons) または (Steps,) の形状
        bin_size (int): 集計するステップ数 (デフォルト10)
        title (str): グラフのタイトル
    """
    # 1次元配列に変換 (ニューロンが複数の場合は合計する)
    if spike_history.ndim > 1:
        data = np.sum(spike_history, axis=1)
    else:
        data = spike_history

    total_steps = len(data)
    n_bins = int(np.ceil(total_steps / bin_size))
    
    print(f"\n{CLIColors.BOLD}=== {title} (Bin Size: {bin_size} steps) ==={CLIColors.RESET}")
    print(f"Legend: {CLIColors.GRAY}0{CLIColors.RESET} < "
          f"{CLIColors.BLUE}1-2{CLIColors.RESET} < "
          f"{CLIColors.GREEN}3-5{CLIColors.RESET} < "
          f"{CLIColors.YELLOW}6-9{CLIColors.RESET} < "
          f"{CLIColors.RED}10+{CLIColors.RESET}")
    
    output_buffer = ""
    
    for i in range(n_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, total_steps)
        
        # 区間内の発火総数をカウント
        count = int(np.sum(data[start:end]))
        
        # 表示文字と色の決定
        char = str(count) if count < 10 else "+" # 10以上は '+' で表現
        
        if count == 0:
            color = CLIColors.GRAY
            char = "_" # ゼロは見やすくアンダースコアに
        elif count <= 2:
            color = CLIColors.BLUE
        elif count <= 5:
            color = CLIColors.GREEN
        elif count <= 9:
            color = CLIColors.YELLOW
        else:
            color = CLIColors.RED + CLIColors.BOLD
            
        output_buffer += f"{color}{char}{CLIColors.RESET}"
        
        # 100ビン (1000ステップ相当) ごとに改行
        if (i + 1) % 100 == 0:
            output_buffer += "\n"

    print(output_buffer)
    print(f"{CLIColors.GRAY}{'-'*50}{CLIColors.RESET}")