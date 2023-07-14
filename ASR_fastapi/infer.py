# from speechbrain.pretrained import WhisperASR
# from hyperpyyaml import load_hyperpyyaml
# import speechbrain as sb
# import torch
# from transformers.models.whisper.tokenization_whisper import LANGUAGES
# import torchaudio
# import torchaudio.transforms as T
# import os

# # Defining tokenizer and loading it
# # hparams_file, run_opts, overrides = sb.parse_arguments(["/home/farah/Desktop/ASR_inf/xheckpoint/hyperparams.yaml"])

# # HPARAMS_NEEDED = ["tokenizer"]
# # MODULES_NEEDED = [
# #     "valid_greedy_searcher",
# #     "test_beam_searcher",
# # ]
# asr_model = WhisperASR.from_hparams(source="/home/farah/Desktop/ASR_infer/ar/save/CKPT+2023-05-24+15-09-00+00", hparams_file='hyperparams1.yaml', savedir="pretrained_model")
# # print(asr_model.transcribe_file('/home/farah/Desktop/ASR_inf/data/splits/test/03BD00C0-2C0B-4C81-BA8C-018175D0B4E3-1.wav'))
# def treat_wav_file(file_mic, file_upload,asr=asr_model, device="cpu") :

    
    
#     if (file_mic is not None) and (file_upload is not None):
#        warn_output = "WARNING: You've uploaded an audio file and used the microphone. The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
#        wav = file_mic
#     elif (file_mic is None) and (file_upload is None):
#        return "ERROR: You have to either use the microphone or upload an audio file"
#     elif file_mic is not None:
#        wav = file_mic
#     else:
#        wav = file_upload
#     sig, sr = torchaudio.load(wav)
#     tensor_wav = sig.to(device)
#     if sr != 16000:
#         resampled = T.Resample(sr, 16000, dtype=torch.float)(tensor_wav)
#     else:
#         resampled=tensor_wav
#     resampled_wav = resampled.squeeze().cpu().detach().numpy()
#     torchaudio.save('resampled.wav', torch.from_numpy(resampled_wav).unsqueeze(0), 16000)
#     resampled_wav='resampled.wav'
#     #resampled = resampled.unsqueeze(0)

#     list_transc=asr_model.transcribe_file(resampled_wav)

#     sentence = ' '
#     for i in range (len(list_transc[0])):
#         sentence = sentence + ' '+str(list_transc[0][i])
#     # file_path="/home/farah/Desktop/ASR_inf/resampled.wav"
#     # if os.path.exists(file_path):
#     #     # Delete the file
#     #     os.remove(file_path)
#     #     print("File deleted successfully.")
#     # else:
#     #     print("File does not exist.")
    
#     # return sentence

#     from transformers import BertForSequenceClassification
#     from transformers import BertTokenizer
#     Model_Used = "UBC-NLP/MARBERT"
#     model = BertForSequenceClassification.from_pretrained(Model_Used, num_labels=3)

#     # Load the trained model from disk
#     model_state_dict = torch.load('/home/farah/Desktop/ASR_infer/marbert_80.pth',
#                                 map_location=torch.device('cpu'))
#     model.load_state_dict(model_state_dict)
#     text = sentence

#     # Tokenize the input text
#     tokenizer = BertTokenizer.from_pretrained(Model_Used)
#     inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]
#     # Forward pass
#     with torch.no_grad():
#         outputs = model(input_ids, attention_mask=attention_mask)

#     # Get the predicted labels
#     predicted_labels = torch.argmax(outputs.logits, dim=1)
#     # Assuming you have a list of class labels ["Label 0", "Label 1", "Label 2"]
#     class_labels = ["Normal", "Abusive", "Discrimination"]

#     # Print the predicted label
#     predicted_label = class_labels[predicted_labels.item()]
#     print("Predicted label:", predicted_label)
#     output='{}:{}'.format(sentence,str(predicted_label))
#     return output

# import gradio as gr
# import torchaudio.transforms as transforms

# title = "My Interface Title"
# description = "My Interface Description"
# with torch.no_grad():
#     gr.Interface(
#         fn=treat_wav_file, 
#         title = title, 
#         description = description,
#         inputs=[gr.inputs.Audio(source="microphone", type='filepath', optional=True),
#             gr.inputs.Audio(source="upload", type='filepath', optional=True)]
#         ,outputs="text").launch()
