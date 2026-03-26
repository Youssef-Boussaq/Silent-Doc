import os
import sys
import whisper
import speech_recognition as sr
from google import genai
from gtts import gTTS

# --- مكتبات إصلاح اللغة العربية فـ الـ Terminal ---
import arabic_reshaper
from bidi.algorithm import get_display

def print_arabic(text):
    """هادي دالة كتقاد الحروف وتلاصقهم باش يتقراو مزيان فـ الويندوز"""
    try:
        reshaped_text = arabic_reshaper.reshape(text)    # تلاصق الحروف
        bidi_text = get_display(reshaped_text)           # تقلب الاتجاه من اليمين لليسار
        print(bidi_text)
    except:
        print(text) # إلى وقع شي مشكل، طبعو عادي

# 1. إعداد مسار FFmpeg 
os.environ["PATH"] += os.path.pathsep + os.getcwd()

# 2. إعداد Gemini API
client = genai.Client(api_key="")

# 3. تحميل محرك Whisper
print_arabic("--- جاري تحميل محرك الذكاء الاصطناعي للصوت (Whisper)... ---")
whisper_model = whisper.load_model("base")

def voice_consultation():
    print("\n" + "="*50)
    print_arabic(" 🩺 مرحبا بك في Silent-Doc (النسخة الصوتية) ")
    print("="*50)
    
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print_arabic("\n[Silent-Doc]: راني كنسمعك، شنو هو المشكل اللي عندك؟ (هضر بالدارجة دابا)")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        
        with open("input.wav", "wb") as f:
            f.write(audio.get_wav_data())

    print_arabic("--- جاري تحليل الكلام... ---")
    try:
        # 4. تحويل الصوت ديالك لنص
        result = whisper_model.transcribe("input.wav", language="ar")
        user_speech = result["text"].strip()
        
        if not user_speech:
            print_arabic("❌ ماسمعت والو، تأكد باللي الميكروفون خدام.")
            return

        print_arabic(f"\n🗣️ أنت قلتي: {user_speech}")

        # 5. استشارة Gemini
        print_arabic("--- جاري استشارة الطبيب الرقمي... ---")
        sys_prompt = "أنت طبيب مغربي رقمي. أجب بالدارجة المغربية فقط وبشكل مختصر جدا ومهني. المريض يقول: "
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=sys_prompt + user_speech
        )
        
        doctor_reply = response.text
        print_arabic(f"\n👨‍⚕️ [Silent-Doc]: {doctor_reply}\n")

        # 6. تحويل الجواب لصوت
        tts = gTTS(text=doctor_reply, lang='ar')
        tts.save("output.mp3")
        os.system("start output.mp3")
        
    except Exception as e:
        print_arabic(f"❌ وقع خطأ تقني: {e}")

if __name__ == "__main__":
    voice_consultation()