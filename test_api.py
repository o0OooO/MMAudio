"""
MMAudio API 测试脚本
"""
import requests
import time
from pathlib import Path

# API 基础 URL
BASE_URL = "http://localhost:8000"


def test_health():
    """测试健康检查"""
    print("\n=== 测试健康检查 ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    return response.status_code == 200


def test_list_models():
    """测试列出模型"""
    print("\n=== 测试列出模型 ===")
    response = requests.get(f"{BASE_URL}/api/v1/models")
    print(f"状态码: {response.status_code}")
    data = response.json()
    print(f"可用模型: {data['available_models']}")
    return response.status_code == 200


def test_text_to_audio():
    """测试文生音频"""
    print("\n=== 测试文生音频 ===")
    
    data = {
        "prompt": "a dog barking loudly",
        "negative_prompt": "",
        "duration": 8.0,
        "cfg_strength": 4.5,
        "num_steps": 25,
        "seed": 42,
        "model_variant": "large_44k_v2"
    }
    
    print(f"请求参数: {data}")
    print("正在生成音频...")
    
    response = requests.post(f"{BASE_URL}/api/v1/text-to-audio", json=data)
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"任务ID: {result['task_id']}")
        print(f"音频URL: {result['audio_url']}")
        
        # 下载音频
        audio_url = f"{BASE_URL}{result['audio_url']}"
        audio_response = requests.get(audio_url)
        
        if audio_response.status_code == 200:
            output_path = Path("test_output_text.flac")
            with open(output_path, "wb") as f:
                f.write(audio_response.content)
            print(f"✓ 音频已保存到: {output_path}")
            return True
        else:
            print(f"✗ 下载音频失败: {audio_response.status_code}")
            return False
    else:
        print(f"✗ 生成失败: {response.text}")
        return False


def test_video_to_audio(video_path):
    """测试视频生音频"""
    print("\n=== 测试视频生音频 ===")
    
    if not Path(video_path).exists():
        print(f"✗ 视频文件不存在: {video_path}")
        print("跳过视频生音频测试")
        return False
    
    with open(video_path, "rb") as video_file:
        files = {"video": video_file}
        data = {
            "prompt": "ocean waves crashing",
            "duration": "8.0",
            "cfg_strength": "4.5",
            "num_steps": "25",
            "seed": "42",
            "model_variant": "large_44k_v2"
        }
        
        print(f"上传视频: {video_path}")
        print("正在生成音频...")
        
        response = requests.post(
            f"{BASE_URL}/api/v1/video-to-audio",
            files=files,
            data=data
        )
    
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"任务ID: {result['task_id']}")
        print(f"音频URL: {result['audio_url']}")
        
        if 'video_url' in result:
            print(f"视频URL: {result['video_url']}")
        
        # 下载音频
        audio_url = f"{BASE_URL}{result['audio_url']}"
        audio_response = requests.get(audio_url)
        
        if audio_response.status_code == 200:
            output_path = Path("test_output_video.flac")
            with open(output_path, "wb") as f:
                f.write(audio_response.content)
            print(f"✓ 音频已保存到: {output_path}")
            return True
        else:
            print(f"✗ 下载音频失败: {audio_response.status_code}")
            return False
    else:
        print(f"✗ 生成失败: {response.text}")
        return False


def test_system_info():
    """测试系统信息"""
    print("\n=== 测试系统信息 ===")
    response = requests.get(f"{BASE_URL}/api/v1/system/info")
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"设备: {data['device']}")
        print(f"PyTorch 版本: {data['torch_version']}")
        if data.get('gpu_info'):
            print(f"GPU 信息: {data['gpu_info']}")
        return True
    return False


def test_system_stats():
    """测试系统统计"""
    print("\n=== 测试系统统计 ===")
    response = requests.get(f"{BASE_URL}/api/v1/system/stats")
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"CPU 使用率: {data['cpu']['percent']}%")
        print(f"内存使用率: {data['memory']['percent']}%")
        print(f"磁盘使用率: {data['disk']['percent']}%")
        if 'gpu' in data:
            print(f"GPU 显存使用: {data['gpu']['memory_allocated_gb']:.2f} GB")
        return True
    return False


def test_list_tasks():
    """测试列出任务"""
    print("\n=== 测试列出任务 ===")
    response = requests.get(f"{BASE_URL}/api/v1/tasks")
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"任务总数: {data['total']}")
        if data['tasks']:
            print("最近的任务:")
            for task in data['tasks'][:3]:
                print(f"  - {task['task_id']}: {task['file_count']} 个文件")
        return True
    return False


def run_all_tests(video_path=None):
    """运行所有测试"""
    print("=" * 60)
    print("MMAudio API 测试")
    print("=" * 60)
    
    results = {}
    
    # 基础测试
    results['health'] = test_health()
    results['models'] = test_list_models()
    results['system_info'] = test_system_info()
    results['system_stats'] = test_system_stats()
    
    # 功能测试
    results['text_to_audio'] = test_text_to_audio()
    
    if video_path:
        results['video_to_audio'] = test_video_to_audio(video_path)
    
    results['list_tasks'] = test_list_tasks()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:20s}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
    
    return passed == total


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 MMAudio API")
    parser.add_argument("--video", type=str, help="测试视频文件路径（可选）")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000",
                       help="API 基础 URL")
    
    args = parser.parse_args()
    BASE_URL = args.base_url
    
    print(f"API 地址: {BASE_URL}")
    print("确保 API 服务正在运行...")
    time.sleep(1)
    
    success = run_all_tests(args.video)
    exit(0 if success else 1)
