import os
from fastapi import FastAPI, UploadFile, File
import httpx
import uvicorn
import requests
import tempfile
from pydantic import BaseModel
from aiohttp import ClientResponse, FormData, ClientSession, ClientConnectorError, ContentTypeError
app = FastAPI()

asr_microservice_url = os.getenv("ASR_MICROSERVICE_URL", "http://127.0.0.1:9012")
classification_microservice_url = os.getenv("CLASSIFICATION_MICROSERVICE_URL", "http://127.0.0.1:8001")


class TextInput(BaseModel):
    text: str

async def file_to_data(payload_obj) -> FormData:
    """
    Args:
        payload_obj: convert file to aio http form data so it can be send in the request
    Returns: aiohttp FormData that could be used on async methods
    """
    temp = tempfile.NamedTemporaryFile(mode="w+b", delete=False)
    temp.name = payload_obj.filename
    data = FormData()
    try:
        temp.writelines(payload_obj.file)
        temp.seek(0)
        data.add_field('wav', temp.read(), filename=payload_obj.filename)
        temp.close()
    except Exception as exception:
      print("hi")
    return data

# @app.post("/transcribe")
# async def transcribe(audio_file: UploadFile = File(..., media_type="audio/wav"), device="cpu"):
#     # async with httpx.AsyncClient() as client:
#     #     asr_response = await client.post(
#     #         f"{asr_microservice_url}/transcribe", files={"wav": audio_file.file}
#     #     )
#     #     transcription = asr_response.json().get("Transcription")
#     #     print(transcription)

#     async with ClientSession() as session:
#         request = getattr(session, "post")
#         async with request(
#                 url=f"{asr_microservice_url}/transcribe",
#                 data=await file_to_data(audio_file),
#         ) as response:

#             return await response.json()
@app.post("/transcribe")

async def transcribe(audio_file: UploadFile = File(..., media_type="audio/wav"), device="cpu"):
    async with ClientSession() as session1:
        async with session1.post(
            url=f"{asr_microservice_url}/transcribe",
            data=await file_to_data(audio_file),
        ) as response:
            transcription = await response.json()
    
    async with ClientSession() as session2:
        async with session2.post(
            url=f"{classification_microservice_url}/classify",
            json={"text": transcription.get("Transcription")},
        ) as response1:
            label = await response1.json()

    return {"transcription": transcription["Transcription"], "label": label["label"]}


        
@app.post('/classifytext')
def classify_text(text_input: dict):
    try:
        response = requests.post(f"{classification_microservice_url}/classify", json=text_input)
        response.raise_for_status()
        result = response.text
        return result
    except requests.exceptions.RequestException as e:
        # Handle errors from the classification microservice
        print("Error: ", e)
        return None
    # async with httpx.AsyncClient() as client:
    #     classification_response =  await client.post(
    #         classification_microservice_url, data=text.encode("utf-8")
    #     )
    #     classification = classification_response.text

    # return {"classification": classification}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=9010)
