import mujoco
import mujoco.viewer
import sys
import time

if len(sys.argv) < 2:
    print("사용법: python3 debug_action_viewer.py <XML 파일 경로>")
    sys.exit(1)

xml_path = sys.argv[1]
try:
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    print("모델 로드 성공. 첫 번째 관절에 미세한 힘(0.01)을 가합니다.")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    sys.exit(1)

with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        # 다른 모든 제어는 0으로 두고, 첫 번째 액추에이터에만 작은 힘을 가함
        d.ctrl[:] = 0.0
        d.ctrl[0] = 0.5  # <--- 이 부분이 핵심

        mujoco.mj_step(m, d)
        v.sync()
        time.sleep(m.opt.timestep)
