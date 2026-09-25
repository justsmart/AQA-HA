import argparse
import os
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
from matplotlib.patches import Patch

RAW_RESULTS = {
	'pascal07': {
		'AP': ['0.651+0.0', '0.611+0.011', '0.59+0.0', '0.519+0.0', '0.575+0.0', '0.569+0.0', '0.56+0.0', '0.544+0.0'],
		'1-HL': ['0.941+0.0', '0.939+0.002', '0.937+0.0', '0.929+0.0', '0.931+0.0', '0.932+0.0', '0.931+0.0', '0.923+0.0'],
		'1-RL': ['0.889+0.0', '0.872+0.008', '0.858+0.0', '0.815+0.0', '0.839+0.0', '0.849+0.0', '0.842+0.0', '0.824+0.0'],
		'AUC': ['0.901+0.0', '0.894+0.009', '0.873+0.0', '0.837+0.0', '0.867+0.0', '0.864+0.0', '0.857+0.0', '0.844+0.0'],
		'1-OE': ['0.562+0.0', '0.537+0.015', '0.499+0.0', '0.436+0.0', '0.49+0.0', '0.456+0.0', '0.451+0.0', '0.449+0.0'],
		'1-Cov': ['0.845+0.0', '0.826+0.008', '0.804+0.0', '0.764+0.0', '0.803+0.0', '0.800+0.0', '0.795+0.0', '0.774+0.0'],
	},
	'mirflickr': {
		'AP': ['0.693+0.0', '0.655+0.009', '0.638+0.0', '0.571+0.0', '0.617+0.0', '0.614+0.0', '0.615+0.0', '0.603+0.0'],
		'1-HL': ['0.915+0.0', '0.900+0.002', '0.898+0.0', '0.885+0.0', '0.893+0.0', '0.892+0.0', '0.892+0.0', '0.89+0.0'],
		'1-RL': ['0.919+0.0', '0.892+0.003', '0.888+0.0', '0.861+0.0', '0.881+0.0', '0.879+0.0', '0.88+0.0', '0.873+0.0'],
		'AUC': ['0.898+0.0', '0.882+0.004', '0.876+0.0', '0.849+0.0', '0.869+0.0', '0.866+0.0', '0.863+0.0', '0.86+0.0'],
		'1-OE': ['0.755+0.0', '0.707+0.014', '0.688+0.0', '0.603+0.0', '0.667+0.0', '0.662+0.0', '0.657+0.0', '0.651+0.0'],
		'1-Cov': ['0.735+0.0', '0.709+0.004', '0.697+0.0', '0.659+0.0', '0.689+0.0', '0.687+0.0', '0.685+0.0', '0.662+0.0'],
	},
}


METRICS_TO_PLOT = ['AP', 'AUC', '1-OE', '1-Cov']
COLOR_STYLES = {
	'candy': {
		0.0: '#5ec2b7',
		0.3: '#f3a35c',
		0.5: '#ef7f84',
		0.7: '#7fa69a',
	},
	'macarons': {
		0.0: '#7fc8a9',
		0.3: '#ffd166',
		0.5: '#f4978e',
		0.7: '#90a8c3',
	},
	'earthy': {
		0.0: '#6db1a7',
		0.3: '#d8a15d',
		0.5: '#c97c6d',
		0.7: '#8c9b6e',
	},
	'cool': {
		0.0: '#63b8c6',
		0.3: '#8fa6ff',
		0.5: '#f28f8f',
		0.7: '#87b88f',
	},
}

HATCH_STYLES = {
	'slashes': '////',
	'horizontal': '----',
	'vertical': '||||',
	'dots': '....',
	'cross': 'xxxx',
	'waves': 'ooo',
	'none': None,
}


def ensure_dir(path):
	if not osp.exists(path):
		os.makedirs(path)


def parse_mean_only(value):
	return float(str(value).split('+')[0])


def build_dataset_values(dataset_name):
	dataset_dict = RAW_RESULTS[dataset_name]
	parsed = {}
	for metric_name, values in dataset_dict.items():
		parsed[metric_name] = np.array([parse_mean_only(item) for item in values], dtype=np.float32)
	return parsed


def get_metric_ylim(values):
	min_val = float(np.min(values))
	max_val = float(np.max(values))
	span = max_val - min_val
	padding = max(0.025, span * 0.35)
	lower = max(0.0, min_val - padding)
	upper = min(1.0, max_val + padding * 0.8)
	if upper <= lower:
		upper = min(1.0, lower + 0.1)
	return lower, upper


def prettify_dataset_name(dataset_name):
	if dataset_name.lower() == 'mirflickr':
		return 'MIRFLICKR'
	if dataset_name.lower() == 'pascal07':
		return 'Pascal07'
	return dataset_name


def add_sticker_effect(bar_container):
	for patch in bar_container.patches:
		patch.set_path_effects([
			patheffects.SimplePatchShadow(
				offset=(1.0, -1.0),
				alpha=0.16,
				shadow_rgbFace='#b9ab98'
			),
			patheffects.Normal(),
		])


def add_bar_highlight(ax, y, width, height):
	ax.barh(
		y=y - height * 0.18,
		width=width * 0.985,
		height=height * 0.34,
		left=0,
		color=(1, 1, 1, 0.22),
		edgecolor='none',
		zorder=4,
	)


def plot_metric_bar_panel(ax, dataset_values, metric_name, rate_colors, hatch_style):
	ax.set_facecolor('#f8efe3')

	values = dataset_values[metric_name]
	fixed_lmr_values = values[[0, 1, 2, 3]]
	fixed_vmr_values = values[[4, 5, 6, 7]]
	missing_rates = [0.0, 0.3, 0.5, 0.7]
	ymin, ymax = get_metric_ylim(values)

	group_centers = np.array([0.00, 0.86], dtype=np.float32)
	bar_width = 0.16
	inner_gap = 0.02
	step = bar_width + inner_gap
	offsets = np.array([
		-1.5 * step,
		-0.5 * step,
		 0.5 * step,
		 1.5 * step,
	], dtype=np.float32)

	for idx, missing_rate in enumerate(missing_rates):
		left_center = group_centers[0] + offsets[idx]
		right_center = group_centers[1] + offsets[idx]

		bars_left = ax.bar(
			left_center,
			fixed_lmr_values[idx],
			width=bar_width,
			color=rate_colors[missing_rate],
			edgecolor='#5b5148',
			linewidth=1.85,
			zorder=3,
			alpha=0.98,
			label=f'{missing_rate:g}' if metric_name == METRICS_TO_PLOT[0] else None,
		)

		bars_right = ax.bar(
			right_center,
			fixed_vmr_values[idx],
			width=bar_width,
			color=rate_colors[missing_rate],
			edgecolor='#5b5148',
			linewidth=1.85,
			zorder=3,
			alpha=0.98,
		)

		# 轻微纹理
		if hatch_style is not None:
			ax.bar(
				left_center,
				fixed_lmr_values[idx],
				width=bar_width * 0.86,
				color='none',
				edgecolor=(0.15, 0.12, 0.10, 0.08),
				linewidth=0.0,
				zorder=4,
				hatch=hatch_style,
			)
			ax.bar(
				right_center,
				fixed_vmr_values[idx],
				width=bar_width * 0.86,
				color='none',
				edgecolor=(0.15, 0.12, 0.10, 0.08),
				linewidth=0.0,
				zorder=4,
				hatch=hatch_style,
			)

		# 柔和阴影
		for patch in list(bars_left.patches) + list(bars_right.patches):
			patch.set_path_effects([
				patheffects.SimplePatchShadow(
					offset=(0.9, -0.9),
					alpha=0.14,
					shadow_rgbFace='#b9ab98'
				),
				patheffects.Normal(),
			])

		# 贴纸高光
		ax.bar(
			left_center,
			fixed_lmr_values[idx] * 0.985,
			bottom=fixed_lmr_values[idx] * 0.72,
			width=bar_width * 0.68,
			color=(1, 1, 1, 0.18),
			edgecolor='none',
			zorder=4,
		)
		ax.bar(
			right_center,
			fixed_vmr_values[idx] * 0.985,
			bottom=fixed_vmr_values[idx] * 0.72,
			width=bar_width * 0.68,
			color=(1, 1, 1, 0.18),
			edgecolor='none',
			zorder=4,
		)

	ax.set_xticks(group_centers)
	ax.set_xticklabels(['missing views', 'missing labels'])

	ax.set_xlabel(metric_name, fontsize=12, fontweight='bold', color='#3f372f')
	ax.set_xlim(group_centers[0] - 0.42, group_centers[1] + 0.42)
	ax.set_ylim(ymin, ymax)

	ax.grid(axis='x', linestyle=':', linewidth=0.7, alpha=0.08, color='#b9a892')
	ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.16, color='#b9a892')

	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_color('#7a6d61')
	ax.spines['bottom'].set_color('#7a6d61')
	ax.spines['left'].set_linewidth(1.0)
	ax.spines['bottom'].set_linewidth(1.0)

	ax.tick_params(axis='y', labelsize=10, colors='#534a42')
	ax.tick_params(axis='x', labelsize=14, pad=7, colors='#4b4138')


def plot_dataset_figure(dataset_name, output_dir, color_style, hatch_style_name):
	dataset_values = build_dataset_values(dataset_name)
	figure_title = prettify_dataset_name(dataset_name)
	rate_colors = COLOR_STYLES[color_style]
	hatch_style = HATCH_STYLES[hatch_style_name]

	plt.rcParams.update({
		'font.family': 'DejaVu Sans',
		'font.size': 11,
		'axes.linewidth': 1.0,
		'hatch.linewidth': 0.3,
		'xtick.direction': 'out',
		'ytick.direction': 'out',
		'legend.frameon': False,
	})

	fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.9))
	fig.patch.set_facecolor('#f3e7d8')

	for ax, metric_name in zip(axes.flatten(), METRICS_TO_PLOT):
		plot_metric_bar_panel(ax, dataset_values, metric_name, rate_colors, hatch_style)

	# fig.suptitle(
	# 	figure_title,
	# 	y=0.965,
	# 	fontsize=16,
	# 	fontweight='bold',
	# 	color='#3f372f'
	# )

	fig.subplots_adjust(
		left=0.10,
		right=0.985,
		bottom=0.09,
		top=0.86,
		wspace=0.18,
		hspace=0.26
	)

	handles, labels = axes.flatten()[0].get_legend_handles_labels()
	legend_handles = [
		Patch(facecolor='none', edgecolor='none', label='missing rate')
	] + handles
	legend_labels = ['missing rate'] + labels
	legend = fig.legend(
		legend_handles,
		legend_labels,
		loc='upper center',
		bbox_to_anchor=(0.5, 0.93),
		ncol=5,
		fontsize=14,
		frameon=True,
		fancybox=True,
		borderpad=0.52,
		handlelength=1.7,
		handletextpad=0.55,
		columnspacing=1.0,
	)

	legend.get_frame().set_facecolor('#fff8ef')
	legend.get_frame().set_edgecolor('#6b5f54')
	legend.get_frame().set_linewidth(1.0)
	legend.get_frame().set_boxstyle('round,pad=0.38,rounding_size=1.0')

	for text in legend.get_texts():
		text.set_color('#2a2623')

	save_path = osp.join(output_dir, f'{dataset_name}_missing_rate_comparison.pdf')
	fig.savefig(save_path, dpi=400, bbox_inches='tight')
	plt.close(fig)
	print(f'Saved figure to {save_path}')


def parse_args():
	parser = argparse.ArgumentParser()
	working_dir = osp.dirname(osp.abspath(__file__))
	parser.add_argument('--output-dir', type=str, default=osp.join(working_dir, 'figs'))
	parser.add_argument('--datasets', nargs='*', default=['pascal07', 'mirflickr'])
	parser.add_argument('--color-style', type=str, default='cool', choices=sorted(COLOR_STYLES.keys()))
	parser.add_argument('--hatch-style', type=str, default='vertical', choices=sorted(HATCH_STYLES.keys()))
	return parser.parse_args()


def main():
	args = parse_args()
	ensure_dir(args.output_dir)
	for dataset_name in args.datasets:
		if dataset_name not in RAW_RESULTS:
			raise ValueError(f'Unsupported dataset: {dataset_name}')
		plot_dataset_figure(dataset_name, args.output_dir, args.color_style, args.hatch_style)


if __name__ == '__main__':
	main()