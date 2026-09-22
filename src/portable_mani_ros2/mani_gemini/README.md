# Gemini Robotics ER simulation adapter

This package is intentionally proposal-only. It sends one MuJoCo RGB frame to
`gemini-robotics-er-2-preview`, validates the structured response, and converts
the model's normalized `[y, x]` point to pixels. It does not publish a ROS
command, write MuJoCo state, or control a physical Dobot.

Install the optional client only in the MuJoCo environment:

```bash
python -m pip install google-genai pillow
export GEMINI_API_KEY='...'
```

Inspect the request without using the API:

```bash
python -m mani_gemini.probe \
  --image mujoco_sim/artifacts/magician_mujoco.png \
  --task 'find the card and propose a safe center point' \
  --dry-run
```

The next integration step is a ROS node that feeds the returned pixel point to
the existing RGB-D localizer and only calls `/mani/pick_object` after a
workspace and confidence safety gate.
