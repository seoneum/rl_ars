#!/usr/bin/env python3
"""
역관절 4족보행 로봇 서기 학습 스크립트
- 앉은 자세에서 시작
- 50~70% 무릎 각도로 서서 밸런스 유지
- 제자리에서 안정적으로 서기 목표
"""

import subprocess
import sys

def train_phase1():
    """Phase 1: 앉은 자세에서 일어서기 학습"""
    print("=" * 60)
    print("Phase 1: 앉은 자세에서 일어서기 학습")
    print("초기 무릎 굴곡: 80%, 목표: 50~70%로 서기")
    print("=" * 60)
    
    cmd = [
        "python", "mjx_ars_train.py",
        "--xml", "quadruped.xml",
        "--save-path", "ars_standing_phase1.npz",
        "--iterations", "300",
        "--num-envs", "256",
        "--num-dirs", "32",
        "--top-dirs", "8",
        "--episode-length", "200",
        "--action-repeat", "3",
        "--step-size", "0.008",
        "--noise-std", "0.012",
        # 초기 자세 설정
        "--crouch-init-ratio", "0.80",  # 깊게 앉은 자세에서 시작
        "--crouch-init-noise", "0.02",
        "--init-pitch", "-0.08",
        # 무릎 목표 범위
        "--knee-band-low", "0.50",
        "--knee-band-high", "0.70",
        "--knee-band-weight", "2.0",  # 무릎 범위 보상 강화
        "--knee-center", "0.60",
        "--knee-center-weight", "0.8",
        "--knee-over-penalty", "0.15",
        "--knee-under-penalty", "0.08",
        # 서기 관련 설정
        "--target-z-low", "0.42",  # 초기엔 낮은 높이 허용
        "--target-z-high", "0.52",
        "--z-threshold", "0.22",
        "--stand-bonus", "0.60",
        "--stand-shape-weight", "2.5",
        "--upright-min", "0.80",
        # 움직임 제한
        "--target-speed", "0.0",
        "--overspeed-weight", "3.0",
        "--speed-weight", "0.0",
        # 페널티 설정
        "--tilt-penalty-weight", "0.08",
        "--angvel-penalty-weight", "0.08",
        "--base-vel-penalty-weight", "0.12",
        "--act-delta-weight", "0.001",
        "--ctrl-cost-weight", "0.0002",
        # 연속 서기 보상
        "--streak-weight", "0.08",
        "--streak-scale", "25.0",
    ]
    
    subprocess.run(cmd)
    
def train_phase2():
    """Phase 2: 안정적인 서기 유지 학습 (이전 학습 재개)"""
    print("\n" + "=" * 60)
    print("Phase 2: 안정적인 서기 유지 학습")
    print("이전 학습 결과에서 재개, 더 엄격한 조건으로")
    print("=" * 60)
    
    import os
    import shutil
    
    # Phase 1 체크포인트를 Phase 2로 복사 (첫 실행시)
    if os.path.exists("ars_standing_phase1.npz") and not os.path.exists("ars_standing_phase2.npz"):
        shutil.copy("ars_standing_phase1.npz", "ars_standing_phase2.npz")
        print("Phase 1 체크포인트를 Phase 2로 복사했습니다.")
    
    cmd = [
        "python", "mjx_ars_train.py",
        "--xml", "quadruped.xml",
        "--save-path", "ars_standing_phase2.npz",
        "--resume",  # 이전 학습 재개 (이제 phase2.npz를 이어서 학습)
        "--iterations", "400",
        "--num-envs", "256",
        "--num-dirs", "32",
        "--top-dirs", "8",
        "--episode-length", "250",
        "--action-repeat", "3",
        "--step-size", "0.006",
        "--noise-std", "0.010",
        # 초기 자세 설정 (여전히 앉은 자세에서 시작)
        "--crouch-init-ratio", "0.75",  # 조금 덜 앉은 자세
        "--crouch-init-noise", "0.02",
        "--init-pitch", "-0.06",
        # 무릎 목표 범위 (유지)
        "--knee-band-low", "0.50",
        "--knee-band-high", "0.70",
        "--knee-band-weight", "2.5",
        "--knee-center", "0.60",
        "--knee-center-weight", "1.0",
        "--knee-over-penalty", "0.20",
        "--knee-under-penalty", "0.10",
        # 서기 관련 설정 (더 엄격하게)
        "--target-z-low", "0.45",
        "--target-z-high", "0.55",
        "--z-threshold", "0.25",
        "--stand-bonus", "0.80",
        "--stand-shape-weight", "3.0",
        "--upright-min", "0.85",
        # 움직임 제한 (더 엄격하게)
        "--target-speed", "0.0",
        "--overspeed-weight", "4.0",
        "--speed-weight", "0.0",
        # 페널티 설정 (더 엄격하게)
        "--tilt-penalty-weight", "0.10",
        "--angvel-penalty-weight", "0.10",
        "--base-vel-penalty-weight", "0.15",
        "--act-delta-weight", "0.0015",
        "--ctrl-cost-weight", "0.0003",
        # 연속 서기 보상 (강화)
        "--streak-weight", "0.12",
        "--streak-scale", "20.0",
    ]
    
    subprocess.run(cmd)

def test_policy():
    """학습된 정책 테스트"""
    print("\n" + "=" * 60)
    print("학습된 정책 테스트")
    print("=" * 60)
    
    import os
    
    # 가장 최근 정책 파일 찾기
    policy_files = [
        "ars_standing_phase2.npz",
        "ars_standing_phase1.npz",
        "ars_policy1.npz"
    ]
    
    policy_file = None
    for f in policy_files:
        if os.path.exists(f):
            policy_file = f
            break
    
    if policy_file:
        print(f"정책 파일 사용: {policy_file}")
        cmd = [
            "python", "mjx_ars_train.py",
            "--xml", "quadruped.xml",
            "--save-path", policy_file,
            "--infer",
            "--num-envs", "64",
            "--episode-length", "300"
        ]
        subprocess.run(cmd)
    else:
        print("학습된 정책 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "phase1":
            train_phase1()
        elif sys.argv[1] == "phase2":
            train_phase2()
        elif sys.argv[1] == "test":
            test_policy()
        else:
            print("Usage: python train_standing.py [phase1|phase2|test]")
    else:
        # 기본: phase1 학습
        train_phase1()
        print("\n첫 번째 단계 학습 완료!")
        print("다음 명령어로 계속하세요:")
        print("  python train_standing.py phase2  # 안정화 학습")
        print("  python train_standing.py test    # 테스트")