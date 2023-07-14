# 1. Library imports
import uvicorn ##ASGI
from fastapi import FastAPI
from fastapi import File, UploadFile
from speechbrain.pretrained import WhisperASR
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
import torch
from transformers.models.whisper.tokenization_whisper import LANGUAGES
import torchaudio
import torchaudio.transforms as T

import sys





# 2. Create the app object
#path="/ASR_fastapi/app/whisper"  #path in docker

path = "whisper" #path locally
print(path)
app = FastAPI()
print(sys.path[0])

asr_model = WhisperASR.from_hparams(source=path, savedir=path)

# # 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello from ASR'}

# 3. Expose the prediction functionality, make a prediction from the passed text
#    and return the predicted label with the confidence
@app.post('/transcribe')
async def asr(wav: UploadFile = File(..., media_type="audio/wav"),device="cpu"):

    
    # temp_file_path = os.path.join(sys.path[0], "resampled.wav")
    # print(temp_file_path)

    # with open(temp_file_path, "wb") as temp_file:
    #         temp_file.write(wav.file.read())
    print(wav.filename)
    sig, sr = torchaudio.load(wav.file)
    tensor_wav = sig.to(device)
    if sr != 16000:
        resampled = T.Resample(sr, 16000, dtype=torch.float)(tensor_wav)
    else:
        resampled=tensor_wav
    resampled_wav = resampled.squeeze().cpu().detach().numpy()
    torchaudio.save('resampled.wav', torch.from_numpy(resampled_wav).unsqueeze(0), 16000)
    resampled_wav='resampled.wav'
    list_transc=asr_model.transcribe_file(resampled_wav)

    sentence = ' '
    for i in range (len(list_transc[0])):
        word=''.join(list_transc[0][i].split(" "))
        sentence = sentence +" "+str(word)

    transcription= sentence
    print(transcription)
   # os.remove(temp_file_path)
    return {
        'Transcription': transcription
    }
# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload