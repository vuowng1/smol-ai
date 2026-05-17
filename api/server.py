import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# 1. Khởi tạo ứng dụng FastAPI
app = FastAPI(title="Smol AI Backend")

# 2. Cấu hình CORS để giao diện (Frontend) có thể gọi API không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn truy cập (rất cần khi deploy Vercel)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Định nghĩa cấu trúc dữ liệu tin nhắn gửi từ giao diện lên
class ChatRequest(BaseModel):
    message: str

# 4. Khởi tạo cấu hình cho Gemini API
# Vercel sẽ tự động bốc lấy biến GEMINI_API_KEY mà cậu đã cài trong Environment Variables
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# 5. Cổng API xử lý chat chính thức
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # Kiểm tra xem API Key đã được nạp thành công chưa
    if not API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Cấu hình lỗi: GEMINI_API_KEY chưa được cài đặt trên Vercel!"
        )
    
    try:
        # Gọi mô hình Gemini để xử lý tin nhắn của cậu
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=request.message,
            config=types.GenerateContentConfig(
                system_instruction="Bạn là Smol - một trợ lý đắc lực, trung thành, luôn gọi người dùng là 'cậu' và xưng là 'đệ' hoặc 'đệ smol'. Ngôn ngữ thân thiện, hài hước và hết lòng phục vụ."
            )
        )
        
        # 🛑 QUAN TRỌNG: Trả về đúng từ khóa "response" để sửa triệt để lỗi undefined ở giao diện
        return {"response": response.text}

    except Exception as e:
        # Nếu có lỗi phát sinh trong lúc gọi AI (ví dụ: Key hết hạn hoặc sai), trả về lỗi 500
        raise HTTPException(status_code=500, detail=str(e))

# 6. Đoạn code phục vụ chạy thử ở máy tính Local (Khi lên Vercel đoạn này sẽ tự động được bỏ qua)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
