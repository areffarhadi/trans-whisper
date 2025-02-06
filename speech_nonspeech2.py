import torch
from speechbrain.pretrained import VAD
import torchaudio
import os
from tqdm import tqdm
import tempfile
import shutil

def process_audio_files(folder_path, threshold=0.5):
    """
    Process all WAV files in a folder using SpeechBrain's pre-trained VAD model
    
    Args:
        folder_path (str): Path to folder containing WAV files
        threshold (float): Classification threshold (0-1), higher means stricter speech detection
    
    Returns:
        dict: Results for each file
    """
    # Create a temporary directory for model files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Loading pre-trained model...")
        # Load pre-trained VAD model using temporary directory
        VAD_model = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir=temp_dir
        )
        
        results = {}
        
        # Process each WAV file
        for filename in tqdm(os.listdir(folder_path)):
            if filename.lower().endswith('.wav'):
                file_path = os.path.join(folder_path, filename)
                try:
                    # Get speech probability scores
                    speech_prob = VAD_model.get_speech_prob_file(file_path)
                    
                    # Calculate average speech probability
                    avg_speech_prob = float(speech_prob.mean())
                    
                    # Classify as speech if average probability exceeds threshold
                    is_speech = avg_speech_prob > threshold
                    
                    results[filename] = {
                        'is_speech': is_speech,
                        'speech_probability': avg_speech_prob,
                    }
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    results[filename] = {'error': str(e)}
    
    return results

def save_results(results, output_file):
    """
    Save results to a text file
    """
    with open(output_file, 'w') as f:
        f.write("Filename,Classification,Speech Probability\n")
        for filename, result in results.items():
            if 'error' in result:
                f.write(f"{filename},ERROR,{result['error']}\n")
            else:
                classification = "Speech" if result['is_speech'] else "Non-speech"
                probability = result['speech_probability']
                f.write(f"{filename},{classification},{probability:.4f}\n")

def main():
    # Replace with your folder path
    folder_path = "only_whispered"
    
    # Output file path
    output_file = os.path.join(folder_path, "speech_detection_results.csv")
    
    # Process files
    print("Processing audio files...")
    results = process_audio_files(folder_path)
    
    # Save results to file
    save_results(results, output_file)
    
    # Print results
    print("\nClassification Results:")
    print("-" * 50)
    for filename, result in results.items():
        if 'error' in result:
            print(f"{filename}: Error - {result['error']}")
        else:
            classification = "Speech" if result['is_speech'] else "Non-speech"
            probability = result['speech_probability']
            print(f"{filename}: {classification} (confidence: {probability:.2%})")
    
    print(f"\nResults have been saved to: {output_file}")

if __name__ == "__main__":
    main()
