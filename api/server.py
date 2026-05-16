import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from google import genai
from google.genai import types

app = FastAPI(title="Cracked Gemini by smol", version="2.5 Pro")

# Mở khóa toàn bộ chính sách CORS để liên kết giữa các cổng (5500 và 8000) thông suốt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo SDK Client với API Key Pro của bạn
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# ----------------------------------------------------
# KHAI BÁO KIỂU DỮ LIỆU ĐẦU VÀO TRUYỀN TỪ UI
# ----------------------------------------------------
class MessageItem(BaseModel):
    role: str
    content: str

class AdvancedChatRequest(BaseModel):
    prompt: str
    history: List[MessageItem] = []
    model: str = "gemini-2.5-flash"
    system_instruction: str = ""
    temperature: float = 0.7

# ----------------------------------------------------
# ENDPOINTS ROUTING
# ----------------------------------------------------

@app.get("/")
def index():
    """Trả về giao diện file HTML khi truy cập thẳng cổng 8000"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "online", "msg": "Hãy đặt file index.html cùng thư mục với server.py để kích hoạt giao diện."}


@app.post("/api/v1/chat")
async def advanced_chat(request: AdvancedChatRequest):
    try:
        # Build cấu trúc dữ liệu hội thoại bao gồm lịch sử truyền từ FrontEnd
        contents = []
        for msg in request.history:
            contents.append(
                types.Content(
                    role=msg.role,
                    parts=[types.Part.from_text(text=msg.content)]
                )
            )
        
        # Đẩy câu hỏi hiện tại của người dùng vào cuối luồng xử lý
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=request.prompt)]
            )
        )

        # Áp dụng cấu hình cá nhân hóa xưng hô gửi từ Menu Cài đặt Frontend sang
        config = types.GenerateContentConfig(
            system_instruction=request.system_instruction,
            temperature=request.temperature
        )

        # Gọi API sinh nội dung văn bản từ mô hình Gemini
        response = client.models.generate_content(
            model=request.model,
            contents=contents,
            config=config
        )

        return {
            "status": "success",
            "response": response.text if response.text else "Đệ smol chưa nghĩ ra câu trả lời cho cậu rồi..."
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Khởi chạy server tại cổng 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
