from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "fish_cli.py"
DEFAULT_BASE_MODEL = "yolo11s.pt"
_BOOTSTRAP_ENV = "FISH_STREAMLIT_BOOTSTRAPPED"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".m4v", ".wmv")


def _bootstrap_streamlit_run() -> None:
    if __name__ != "__main__":
        return
    if os.environ.get(_BOOTSTRAP_ENV) == "1":
        return

    try:
        from streamlit.runtime import exists as streamlit_runtime_exists
    except Exception:
        return

    if streamlit_runtime_exists():
        return

    os.environ[_BOOTSTRAP_ENV] = "1"
    script_path = str(Path(__file__).resolve())
    print("Launching Streamlit UI via `streamlit run`...")
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            script_path,
            *sys.argv[1:],
        ],
    )


_bootstrap_streamlit_run()

import streamlit as st


def _resolve_preview_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _find_direct_video_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ]
    )


def _resolve_workflow_input_path(value: str) -> Path:
    path = _resolve_preview_path(value)
    if path.is_dir():
        direct_videos = _find_direct_video_files(path)
        if len(direct_videos) == 1:
            return direct_videos[0]
    return path


def _run_cli(args: list[str]) -> dict[str, object]:
    command = [sys.executable, str(CLI_PATH), *args]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "success": result.returncode == 0,
    }


def _trim_text(value: str, limit: int = 40000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _render_text_block(title: str, content: str, *, expanded: bool) -> None:
    if not content.strip():
        return
    with st.expander(title, expanded=expanded):
        st.code(_trim_text(content))


def _preview_file(path: Path) -> None:
    st.markdown(f"**{path.name}**")
    if not path.exists():
        st.warning(f"Missing artifact: {path}")
        return

    st.caption(str(path))
    suffix = path.suffix.lower()

    if suffix == ".csv":
        try:
            dataframe = pd.read_csv(path, nrows=250)
            st.dataframe(dataframe, use_container_width=True)
        except Exception as exc:
            st.warning(f"Unable to preview CSV: {exc}")
        return

    if suffix == ".json":
        try:
            st.json(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            st.warning(f"Unable to preview JSON: {exc}")
        return

    if suffix == ".mp4":
        st.video(str(path))
        return

    if suffix in {".log", ".txt"}:
        try:
            st.code(_trim_text(path.read_text(encoding="utf-8")))
        except Exception as exc:
            st.warning(f"Unable to preview text file: {exc}")
        return

    st.write("Artifact created.")


def _build_output_paths(video_value: str, output_dir_value: str) -> tuple[Path, Path, Path]:
    video_path = _resolve_workflow_input_path(video_value)
    output_dir = _resolve_preview_path(output_dir_value)
    stem = video_path.stem or "video"
    csv_path = output_dir / f"{stem}_tracks.csv"
    preview_video_path = output_dir / f"{stem}_visualized.mp4"
    return output_dir, csv_path, preview_video_path


def _run_video_workflow(
    video_value: str,
    weights_value: str,
    output_dir_value: str,
) -> dict[str, object]:
    video_path = _resolve_workflow_input_path(video_value)
    output_dir, csv_path, preview_video_path = _build_output_paths(
        video_value,
        output_dir_value,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_args = ["run", str(video_path), str(csv_path)]
    visualize_args = ["visualize", str(video_path), str(preview_video_path)]
    weights_path = weights_value.strip()
    if weights_path:
        tracking_args.extend(["--weights", weights_path])
        visualize_args.extend(["--weights", weights_path])

    steps: list[dict[str, object]] = []
    track_result = _run_cli(tracking_args)
    track_result["title"] = "Tracking"
    steps.append(track_result)

    if track_result["success"]:
        visualize_result = _run_cli(visualize_args)
        visualize_result["title"] = "Visualization"
        steps.append(visualize_result)

    return {
        "success": all(step["success"] for step in steps),
        "video_path": str(video_path),
        "weights_path": weights_path,
        "output_dir": str(output_dir),
        "tracks_csv": str(csv_path),
        "visualized_video": str(preview_video_path),
        "steps": steps,
    }


def _render_workflow_result() -> None:
    result = st.session_state.get("workflow_result")
    if not result:
        return

    if result["success"]:
        st.success("Finished tracking and visualization.")
    else:
        st.error("The workflow did not complete successfully.")

    st.markdown("### Outputs")
    _preview_file(_resolve_preview_path(str(result["visualized_video"])))
    _preview_file(_resolve_preview_path(str(result["tracks_csv"])))

    st.markdown("### Command Logs")
    for step in result["steps"]:
        st.markdown(f"#### {step['title']}")
        st.caption(subprocess.list2cmdline(step["command"]))
        _render_text_block(
            f"{step['title']} stdout",
            str(step["stdout"]),
            expanded=not step["success"],
        )
        _render_text_block(
            f"{step['title']} stderr",
            str(step["stderr"]),
            expanded=not step["success"],
        )


def main() -> None:
    st.set_page_config(page_title="Fish CLI UI", layout="wide")

    st.title("Fish CLI UI")
    st.caption(
        "Input one video file, a folder containing one video, or a folder of frames to "
        "generate both a tracks CSV and an annotated video."
    )
    st.info(
        "Leave the weights field blank to let `fish_cli.py` resolve the default model weights."
    )

    (workflow_tab,) = st.tabs(["Visualize Video"])

    with workflow_tab:
        with st.form("video_workflow_form"):
            video_input = st.text_input(
                "Input video or frames folder",
                value="",
                placeholder=r"C:\path\to\video.mp4 or C:\path\to\video_folder",
            )
            weights_input = st.text_input("Weights file (optional)", value="")
            output_dir_input = st.text_input(
                "Output folder",
                value=str(REPO_ROOT / "outputs"),
            )
            run_submit = st.form_submit_button("Generate CSV and Video", use_container_width=True)

        if video_input.strip():
            output_dir, csv_path, preview_video_path = _build_output_paths(
                video_input,
                output_dir_input,
            )
            st.caption(
                f"Outputs will be written to `{output_dir}` as "
                f"`{csv_path.name}` and `{preview_video_path.name}`."
            )

        if run_submit:
            if not video_input.strip():
                st.session_state.pop("workflow_result", None)
                st.error("Input video or frames folder is required.")
            else:
                video_path = _resolve_workflow_input_path(video_input)
                if not video_path.exists():
                    st.session_state.pop("workflow_result", None)
                    st.error(f"Input path was not found: {video_path}")
                else:
                    with st.spinner("Running tracking and visualization..."):
                        st.session_state["workflow_result"] = _run_video_workflow(
                            video_input,
                            weights_input,
                            output_dir_input,
                        )

        _render_workflow_result()


if __name__ == "__main__":
    main()
