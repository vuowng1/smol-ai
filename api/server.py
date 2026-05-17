import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

# 1. Khởi tạo FastAPI
app = FastAPI(title="Smol AI Backend")

# 2. Cấu hình CORS để Frontend thoải mái gọi API không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Lấy API Key từ Environment Variables của Vercel
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

@app.post("/api/chat")
async def chat_endpoint(request: dict):  # Nhận dict tổng quát để tránh lỗi lệch cấu hình dữ liệu (422)
    if not API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Cấu hình lỗi: GEMINI_API_KEY chưa được cài đặt trên Vercel!"
        )
    
    try:
        # Tự động tìm tin nhắn dựa trên các key phổ biến mà frontend hay gửi
        user_message = request.get("message") or request.get("prompt") or request.get("text")
        
        # Phòng trường hợp frontend gửi chuỗi hoặc object dị biệt
        if not user_message and request:
            user_message = list(request.values())[0]
            
        if not user_message:
            user_message = "Hello"

        # Gọi mô hình Gemini xử lý dữ liệu
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=str(user_message),
            config=types.GenerateContentConfig(
                system_instruction="Bạn là Smol - một trợ lý đắc lực, trung thành, luôn gọi người dùng là 'cậu' và xưng là 'đệ' hoặc 'đệ smol'. Ngôn ngữ thân thiện, hài hước và hết lòng phục vụ."
            )
        )
        
        # Trả ra cả 2 định dạng response và reply để frontend dùng bản nào cũng đọc được dữ liệu
        return {
            "response": response.text,
            "reply": response.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
