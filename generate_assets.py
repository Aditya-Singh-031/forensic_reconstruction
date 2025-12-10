"""
ASSET GENERATOR for Video Presentation.
Creates:
1. The Pivot Diagram (Original vs New Scope)
2. System Architecture Diagram
3. Results Chart (PSNR/SSIM)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Setup
output_dir = Path("output/video_assets")
output_dir.mkdir(parents=True, exist_ok=True)

# Style settings for "Dark Mode" / Cyberpunk Forensic look
plt.style.use('dark_background')
colors = {'box': '#1e293b', 'text': '#e2e8f0', 'arrow': '#3b82f6', 'highlight': '#ef4444', 'success': '#22c55e'}

def draw_box(ax, x, y, w, h, text, color=colors['box'], text_color=colors['text'], fontsize=10):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  linewidth=2, edgecolor=colors['arrow'], facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', color=text_color, 
            fontsize=fontsize, fontweight='bold', wrap=True)

def create_pivot_diagram():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Original Scope
    draw_box(ax, 1, 4, 3, 1.5, "ORIGINAL SCOPE\n\n- Lip Prints\n- Bite Marks\n- Unimodal", color='#334155')
    
    # The Roadblock
    draw_box(ax, 5, 4, 2, 1.5, "ROADBLOCK\n\n❌ No Reliable\nDatasets", color='#7f1d1d', text_color='#fca5a5')
    
    # The Pivot Arrow
    ax.arrow(4.2, 4.75, 0.6, 0, head_width=0.2, head_length=0.2, fc=colors['arrow'], ec=colors['arrow'])
    ax.arrow(7.2, 4.75, 0.6, 0, head_width=0.2, head_length=0.2, fc=colors['arrow'], ec=colors['arrow'])
    
    # The New Scope
    draw_box(ax, 8, 3, 3, 2.5, "NEW SCOPE (Implemented)\n\n- Multimodal (Voice/Text)\n- Feature Corruption\n- Deep Semantic Inpainting\n- Database Matching", color='#064e3b', text_color='#6ee7b7')
    
    # Connection to implementation
    ax.text(6, 1, "Shifted focus from 'Micro-Features' to 'Holistic Reconstruction'", 
            ha='center', fontsize=12, style='italic', color='#94a3b8')

    plt.title("Project Evolution: The Pivot", fontsize=16, color='white', pad=20)
    plt.savefig(output_dir / "1_project_pivot.png", dpi=300, bbox_inches='tight')
    print("✓ Generated Pivot Diagram")

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Input Layer
    draw_box(ax, 1, 6, 2, 1, "INPUT\n(Voice/Text)", color='#1e293b')
    
    # Process Layer
    draw_box(ax, 4, 6, 2.5, 1, "ATTRIBUTE PARSER\n(NLP / Regex)", color='#1e293b')
    
    # Search Layer
    draw_box(ax, 7.5, 6, 2.5, 1, "DATABASE SEARCH\n(CLIP Embeddings)", color='#1e293b')
    
    # The Core (Corruption/Recon)
    rect = patches.Rectangle((3.5, 1), 7, 3.5, linewidth=2, edgecolor='#ef4444', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.text(7, 4.7, "CORE INNOVATION: RECONSTRUCTION ENGINE", ha='center', color='#ef4444', fontweight='bold')
    
    draw_box(ax, 4, 2.5, 2.5, 1, "CORRUPTION\nENGINE\n(Donor Features)", color='#450a0a')
    draw_box(ax, 7.5, 2.5, 2.5, 1, "U-NET MODEL\n(Attention Gates)", color='#064e3b')
    
    # Output
    draw_box(ax, 11, 4, 2, 1, "FINAL\nRESULT", color='#172554')

    # Arrows
    # Input -> Parser
    ax.arrow(3.1, 6.5, 0.8, 0, head_width=0.15, fc='white', ec='white')
    # Parser -> Search
    ax.arrow(6.6, 6.5, 0.8, 0, head_width=0.15, fc='white', ec='white')
    # Search -> Corruption
    ax.arrow(8.75, 6, 0, -1.3, head_width=0.15, fc='white', ec='white')
    # Corruption -> U-Net
    ax.arrow(6.6, 3, 0.8, 0, head_width=0.15, fc='white', ec='white')
    # U-Net -> Output
    ax.arrow(10.1, 3, 0.8, 1, head_width=0.15, fc='white', ec='white')

    plt.title("System Architecture: End-to-End Pipeline", fontsize=16, color='white', pad=20)
    plt.savefig(output_dir / "2_system_architecture.png", dpi=300, bbox_inches='tight')
    print("✓ Generated Architecture Diagram")

def create_results_chart():
    # Data from your log output
    metrics = ['PSNR (dB)', 'SSIM', 'LPIPS']
    values = [36.32, 0.997, 0.009] # Your actual results
    benchmarks = [30.0, 0.85, 0.10] # Standard benchmarks for "Good" quality
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # PSNR
    ax1.bar(['Standard', 'Ours'], [30.0, 36.32], color=['#475569', '#22c55e'])
    ax1.set_title("PSNR (Higher is Better)", fontsize=12)
    ax1.text(1, 36.32, "36.32 dB", ha='center', va='bottom', fontweight='bold', color='#22c55e')
    
    # SSIM
    ax2.bar(['Standard', 'Ours'], [0.85, 0.997], color=['#475569', '#3b82f6'])
    ax2.set_title("SSIM (Higher is Better)", fontsize=12)
    ax2.text(1, 0.997, "0.997", ha='center', va='bottom', fontweight='bold', color='#3b82f6')
    
    # LPIPS
    ax3.bar(['Standard', 'Ours'], [0.10, 0.009], color=['#475569', '#ef4444'])
    ax3.set_title("LPIPS (Lower is Better)", fontsize=12)
    ax3.text(1, 0.009, "0.009", ha='center', va='bottom', fontweight='bold', color='#ef4444')
    
    plt.suptitle("Quantitative Evaluation Results (Test Set)", fontsize=16)
    plt.savefig(output_dir / "3_results_charts.png", dpi=300, bbox_inches='tight')
    print("✓ Generated Results Chart")

if __name__ == "__main__":
    create_pivot_diagram()
    create_architecture_diagram()
    create_results_chart()