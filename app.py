from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from prog_utils.growth import analyze_growth_image, load_growth_model
from prog_utils.objects import analyze_object_image, load_objects_model
from prog_utils.regions import analyze_rgb_image, load_model


PROJECT_NAME = "AI-Enabled Drone-Based Crop Health Monitoring and Advisory System for Precision Agriculture"
PAGES = [
	"01 · Upload",
	"02 · Growth stage",
	"03 · Healthy & unhealthy regions",
	"04 · Weeds",
]
FILE_TYPES = ["png", "jpg", "jpeg", "tif", "tiff"]

st.set_page_config(page_title=PROJECT_NAME, page_icon="🌾", layout="wide")
st.markdown(
	"""
	<style>
		:root {
			--field-green: #245c43;
			--leaf-green: #719b62;
			--sun-yellow: #e6b84a;
			--soil-ink: #20352b;
			--paper: #f5f7f1;
		}
		.stApp { background: var(--paper); color: var(--soil-ink); }
		[data-testid="stHeader"] { background: transparent; }
		[data-testid="stSidebar"] { background: #e9efe5; border-right: 1px solid #d6e0d2; }
		.project-title {
			max-width: 1100px;
			margin: 0.35rem 0 0.6rem;
			color: var(--field-green);
			font-family: Georgia, 'Times New Roman', serif;
			font-size: clamp(1.8rem, 3.4vw, 3.1rem);
			font-weight: 700;
			line-height: 1.08;
		}
		.page-kicker {
			color: #607367;
			font-size: 0.78rem;
			font-weight: 700;
			text-transform: uppercase;
		}
		.stButton > button[kind="primary"] {
			background: var(--field-green);
			border-color: var(--field-green);
		}
		.stButton > button[kind="primary"]:hover {
			background: #174832;
			border-color: #174832;
		}
		[data-testid="stMetric"] {
			background: #fff;
			border: 1px solid #dce5d9;
			border-radius: 6px;
			padding: 0.8rem 1rem;
		}
		[data-testid="stProgressBar"] > div > div { background: var(--leaf-green); }
		@media (max-width: 700px) {
			.project-title { font-size: 1.8rem; }
		}
	</style>
	""",
	unsafe_allow_html=True,
)

if "workflow_page" not in st.session_state:
	st.session_state.workflow_page = PAGES[0]
if "upload_widget_version" not in st.session_state:
	st.session_state.upload_widget_version = 0

st.markdown(f"<div class='project-title'>{PROJECT_NAME}</div>", unsafe_allow_html=True)
st.caption("Precision agriculture · Field image analysis and decision support")

with st.sidebar:
	st.markdown("### Field workflow")
	selected_page = st.radio(
		"Analysis steps",
		PAGES,
		key="workflow_page",
		label_visibility="collapsed",
	)
	st.progress((PAGES.index(selected_page) + 1) / len(PAGES))
	st.caption(f"Step {PAGES.index(selected_page) + 1} of {len(PAGES)}")
	st.divider()
	st.caption("RGB imagery is used for all analyses. Growth-stage analysis requires 1280 × 960 pixels.")


def read_image(image_bytes, mode="RGB"):
	with Image.open(BytesIO(image_bytes)) as image:
		return image.convert(mode)


def clear_analysis_results():
	for key in ("growth_result", "health_result", "weeds_result"):
		st.session_state.pop(key, None)


def render_upload_page():
	st.markdown("<div class='page-kicker'>Start here</div>", unsafe_allow_html=True)
	st.header("Upload field imagery")
	st.write("Add the drone's RGB field image. NIR and red-band imagery can also be attached for future multispectral analysis.")

	version = st.session_state.upload_widget_version
	rgb_upload = st.file_uploader("RGB image · required", type=FILE_TYPES, key=f"rgb_{version}")
	optional_columns = st.columns(2)
	with optional_columns[0]:
		nir_upload = st.file_uploader("NIR image · optional", type=FILE_TYPES, key=f"nir_{version}")
	with optional_columns[1]:
		red_upload = st.file_uploader("R-band image · optional", type=FILE_TYPES, key=f"red_{version}")

	for key, upload in (("rgb_bytes", rgb_upload), ("nir_bytes", nir_upload), ("red_bytes", red_upload)):
		if upload is not None:
			new_bytes = upload.getvalue()
			if st.session_state.get(key) != new_bytes:
				st.session_state[key] = new_bytes
				clear_analysis_results()

	if st.button("Clear uploaded images", disabled=not st.session_state.get("rgb_bytes")):
		for key in ("rgb_bytes", "nir_bytes", "red_bytes"):
			st.session_state.pop(key, None)
		clear_analysis_results()
		st.session_state.upload_widget_version += 1
		st.rerun()

	rgb_bytes = st.session_state.get("rgb_bytes")
	if not rgb_bytes:
		st.info("Upload an RGB image to unlock the analysis pages.")
		return

	try:
		preview_items = [("RGB field image", read_image(rgb_bytes))]
		if st.session_state.get("nir_bytes"):
			preview_items.append(("NIR", read_image(st.session_state.nir_bytes, "L")))
		if st.session_state.get("red_bytes"):
			preview_items.append(("R band", read_image(st.session_state.red_bytes, "L")))
		st.success(f"RGB image ready · {preview_items[0][1].width} × {preview_items[0][1].height} pixels")
		columns = st.columns(len(preview_items))
		for column, (label, image) in zip(columns, preview_items):
			column.image(image, caption=label, width="stretch")
		if len(preview_items) > 1:
			st.info("Multispectral analysis is under development; optional images are stored for this session.")
	except Exception as error:
		st.error(f"Could not open the uploaded image: {error}")


def render_growth_page():
	st.markdown("<div class='page-kicker'>Crop development</div>", unsafe_allow_html=True)
	st.header("Growth stage")
	st.write("Classify the crop stage across 12 non-overlapping field regions.")
	rgb_bytes = st.session_state.get("rgb_bytes")
	if not rgb_bytes:
		st.warning("Upload an RGB image on the Upload page to continue.")
		return
	try:
		rgb_image = read_image(rgb_bytes)
		if rgb_image.size != (1280, 960):
			st.warning("Growth-stage analysis requires an image that is exactly 1280 × 960 pixels.")
			st.caption(f"Current image: {rgb_image.width} × {rgb_image.height} pixels")
			return
		if st.button("Analyze growth stage", type="primary"):
			with st.spinner("Loading growth-stage model and analyzing regions..."):
				model, device = load_growth_model()
				overlay, results, summary = analyze_growth_image(rgb_image, model, device)
			st.session_state.growth_result = (overlay, results, summary)
		if "growth_result" in st.session_state:
			overlay, results, summary = st.session_state.growth_result
			st.image(overlay, caption="Growth-stage regions", width="stretch")
			metric_columns = st.columns(2)
			metric_columns[0].metric("Most common stage", summary["majority_stage"])
			metric_columns[1].metric("Regions analyzed", summary["region_count"])
			st.write("Stage counts", summary["stage_counts"])
			st.dataframe(results, width="stretch", hide_index=True)
	except Exception as error:
		st.error(f"Could not analyze the growth stage: {error}")


def render_health_page():
	st.markdown("<div class='page-kicker'>Plant condition</div>", unsafe_allow_html=True)
	st.header("Healthy and unhealthy regions")
	st.write("Score overlapping image patches and review the field-level health map and advisory.")
	rgb_bytes = st.session_state.get("rgb_bytes")
	if not rgb_bytes:
		st.warning("Upload an RGB image on the Upload page to continue.")
		return
	try:
		rgb_image = read_image(rgb_bytes)
		if st.button("Analyze field health", type="primary"):
			with st.spinner("Loading health model and analyzing image patches..."):
				model = load_model()
				overlay, patch_results, summary = analyze_rgb_image(np.asarray(rgb_image), model)
			st.session_state.health_result = (overlay, patch_results, summary)
		if "health_result" in st.session_state:
			overlay, patch_results, summary = st.session_state.health_result
			st.image(overlay, caption="RGB health map", width="stretch")
			metric_columns = st.columns(4)
			metric_columns[0].metric("Healthy", f"{summary['healthy_pct']:.1f}%")
			metric_columns[1].metric("At risk", f"{summary['at_risk_pct']:.1f}%")
			metric_columns[2].metric("Unhealthy", f"{summary['unhealthy_pct']:.1f}%")
			metric_columns[3].metric("Mean unhealthy score", f"{summary['severity_index']:.1f}%")
			st.info(summary["action_level"])
			st.caption(f"{summary['total_patches']} patches analyzed")
			st.subheader("Patch results")
			st.dataframe(patch_results, width="stretch", hide_index=True)
			st.download_button(
				"Download patch report (CSV)",
				patch_results.to_csv(index=False).encode("utf-8"),
				file_name="patch_evaluation_report.csv",
				mime="text/csv",
			)
	except Exception as error:
		st.error(f"Could not analyze field health: {error}")


def render_weeds_page():
	st.markdown("<div class='page-kicker'>Foreground mapping</div>", unsafe_allow_html=True)
	st.header("Weed and foreground mask")
	st.write("Generate a binary foreground mask and inspect the highlighted regions against the original image.")
	st.info("The current U-Net produces a binary mask; it does not identify separate weed or object categories.")
	rgb_bytes = st.session_state.get("rgb_bytes")
	if not rgb_bytes:
		st.warning("Upload an RGB image on the Upload page to continue.")
		return
	try:
		rgb_image = read_image(rgb_bytes)
		if st.button("Analyze weeds", type="primary"):
			with st.spinner("Loading U-Net and generating the foreground mask..."):
				model = load_objects_model()
				mask, overlay, summary = analyze_object_image(rgb_image, model)
			st.session_state.weeds_result = (mask, overlay, summary)
		if "weeds_result" in st.session_state:
			mask, overlay, summary = st.session_state.weeds_result
			image_columns = st.columns(2)
			image_columns[0].image(mask, caption="Predicted foreground mask", width="stretch")
			image_columns[1].image(overlay, caption="Foreground overlay", width="stretch")
			st.metric("Detected foreground", f"{summary['foreground_pct']:.2f}%")
	except Exception as error:
		st.error(f"Could not analyze weeds: {error}")


{
	PAGES[0]: render_upload_page,
	PAGES[1]: render_growth_page,
	PAGES[2]: render_health_page,
	PAGES[3]: render_weeds_page,
}[selected_page]()
