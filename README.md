# 🗣️ ASR for Whispered Speech #

Whispered speech is a challenging speaking style for normal ASR models. 

In this repository, we share the fine-tuned version of the OpenAI **`Whisper-large-v2`** model for whispered speech. In this fine-tuning, we used the **WTIMIT** and **CHAINS** datasets.


You can download the [fine-tuned Whisper large v2 model](https://drive.google.com/file/d/1MB8qjPk8lmtECmuKX0qXhlXr9uwmnA0g/view?usp=sharing) for whispered speech recognition.

Results are reported in these papers:

1. [Leveraging Self-Supervised Models for Automatic Whispered Speech Recognition](https://arxiv.org/abs/2407.21211)
2. Submitted to Interspeech2025



__________________________________
# Whisper Vs. Normal speech Classification
Using `test_youtube_WN2.py`, you can utilize a fine-tuned ResNet model to classify audio for Normal and Whisper speaking styles. This model's performance on the WTIMIT dataset is 95.15% accuracy. The model is ready to download (here)[https://drive.google.com/file/d/1rxHqRjt80mGKiTldcaQFUNIy5oceblHA/view?usp=drive_link]. To run this code, you need to make manifest using `manifest_voice.py`. You also need to enter the address of the speech data repository in this code snippet.

We utilized this model to create a whisper speech dataset. In large-scale file manipulation, sometimes the utterances are nonspeech (consisting of music, env sound, etc.), and to remove these non-speech utterances, you can use `speech_nonspeech2.py`. This code is based on SpeechBrain's model.

