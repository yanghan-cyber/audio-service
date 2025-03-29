import requests
import time


BASE_URL = "http://localhost:8080/v1"


def get_voices():
    
    url = f"{BASE_URL}/audio/voices"
    
    # 发送请求
    print(f"发送请求到 {url}")
    start_time = time.time()
    response = requests.get(url)
    processing_time = time.time() - start_time
    print(f"请求处理时间: {processing_time:.2f} 秒")

    # 检查响应状态码
    if response.status_code == 200:
        return response.json()
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return None

def test_speech(voice_id):
    url = f"{BASE_URL}/audio/speech"
    data = {
        "input": "很高兴见到你。今天吃些什么东西哦？",
        "model": "iic/CosyVoice2-0.5B",
        "voice": voice_id,
        "response_format": "wav",
        "instruct": "请用四川话说",
        "speed": 1.0
    }
    # 发送请求
    print(f"发送请求到 {url}")
    print(f"使用说话人ID: {voice_id}")
    start_time = time.time()
    response = requests.post(url, json=data)
    processing_time = time.time() - start_time
    # 检查响应
    if response.status_code == 200:
        print(f"请求成功! 处理时间: {processing_time:.2f}秒")
        
        # 保存音频文件
        output_file = "tts_output.mp3"
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        print(f"已保存音频文件: {output_file}")
        print(f"文件大小: {len(response.content) / 1024:.2f} KB")
        
        return True
    else:
        print(f"请求失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        return False

def test_transcribe():
    input_file = "tts_output.mp3"

    url = f"{BASE_URL}/audio/transcribe"
    files = {"file": open(input_file, "rb")}

    data = {
        "model": "iic/SenseVoiceSmall",
        "language": "auto",
    }
    print(f"发送请求到 {url}")
    start_time = time.time()
    response = requests.post(url, files=files, data=data)
    processing_time = time.time() - start_time
    print(f"请求处理时间: {processing_time:.2f} 秒")

    # 检查响应状态码
    if response.status_code == 200:
        return response.json()
    else:
        print(f"请求失败: {response.status_code}")
        print(f"错误信息: {response.text}")
        return None

if __name__ == "__main__":
    voices = get_voices()
    if voices:
        test_speech(voices[0]["id"])
    text = test_transcribe()
    print(text)
