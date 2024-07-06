from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from Extractive_summarization_new import extractive_summarize
from Text_Summarization_Abstractive import abstractive_summarize

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SummarizationRequest(BaseModel):
    text: str
    method: str

@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    text = request.text
    method = request.method

    if method == 'abstractive':
        summary = abstractive_summarize(text)
    else:
        summary = extractive_summarize(text)

    return {"summary": summary}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
