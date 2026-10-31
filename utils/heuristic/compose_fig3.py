'''
Place per-model Fig 3 panels (plot_quadrant_matrix PDFs) side by side in one vector PDF,
scaled to a common height, plus a PNG preview.

uv run -m utils.heuristic.compose_fig3 --model_names deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B QuixiAI/DeepSeek-R1-AWQ openai/gpt-oss-20b
'''

import argparse
import os
import subprocess

from pypdf import PageObject, PdfReader, PdfWriter, Transformation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model_names', nargs='+', required=True)
    parser.add_argument('--results_dir', default='results/')
    parser.add_argument('--panel', default='heatmap_quadrant_rep10_bins30_sampled.pdf')
    parser.add_argument('--gap', type=float, default=12.0, help='Gap between panels (pt)')
    parser.add_argument('--out', default='figures/fig3_quadrant_panels.pdf')
    args = parser.parse_args()

    pages = [PdfReader(os.path.join(args.results_dir, m, 'figures', args.panel)).pages[0] for m in args.model_names]
    height = max(float(p.mediabox.height) for p in pages)
    scales = [height / float(p.mediabox.height) for p in pages]
    width = sum(float(p.mediabox.width) * s for p, s in zip(pages, scales)) + args.gap * (len(pages) - 1)

    canvas = PageObject.create_blank_page(width=width, height=height)
    x = 0.0
    for page, s in zip(pages, scales):
        canvas.merge_transformed_page(page, Transformation().scale(s, s).translate(x, 0))
        x += float(page.mediabox.width) * s + args.gap
    writer = PdfWriter()
    writer.add_page(canvas)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        writer.write(f)
    subprocess.run(['pdftoppm', '-png', '-r', '150', '-singlefile', args.out, os.path.splitext(args.out)[0]], check=True)
    print(f'Saved {args.out} (+ .png): {", ".join(m.split("/")[-1] for m in args.model_names)}')


if __name__ == '__main__':
    main()
