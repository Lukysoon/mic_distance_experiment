import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import pickle
import librosa
from pathlib import Path
import threading
import queue

from AudioDistance import AudioFeatureExtractor, AudioDistanceClassifier


class RealtimeAudioClassifier:
    """Real-time audio classification with visualization"""

    def __init__(self, model_path='trained_model.pkl',
                 sr=16000,
                 chunk_duration=1.0,
                 buffer_size=10):
        """
        Args:
            model_path: path to saved model
            sr: sample rate
            chunk_duration: duration of each analysis chunk in seconds
            buffer_size: number of recent predictions to keep for visualization
        """
        self.sr = sr
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sr * chunk_duration)
        self.buffer_size = buffer_size

        # Load trained model
        self.load_model(model_path)

        # Audio buffer
        self.audio_queue = queue.Queue()
        self.is_running = False

        # Results buffers for visualization
        self.predictions = deque(maxlen=buffer_size)
        self.confidences = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.probabilities_history = deque(maxlen=buffer_size)

        # Current state
        self.current_prediction = None
        self.current_probabilities = None
        self.time_counter = 0

    def load_model(self, model_path):
        """Load trained model from file"""
        print(f"Loading model from {model_path}...")

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please train and save a model first using AudioDistance.py"
            )

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Extract model components
        self.trained_model = model_data['trained_model']
        self.scaler = model_data['scaler']
        self.class_names = model_data['class_names']
        self.n_classes = len(self.class_names)

        # Get feature importance if available
        self.feature_importance = model_data.get('feature_importance', None)
        self.feature_names = model_data.get('feature_names', None)

        # Recreate feature extractor with saved parameters
        fe_params = model_data['feature_extractor_params']
        self.feature_extractor = AudioFeatureExtractor(
            sr=fe_params['sr'],
            n_mfcc=fe_params['n_mfcc']
        )

        # Current feature values for display
        self.current_features = None

        print(f"Model loaded successfully!")
        print(f"Classes: {self.class_names}")

        if self.feature_importance is not None:
            print(f"Top 5 important features loaded")

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")

        # Add audio data to queue
        self.audio_queue.put(indata.copy())

    def process_audio(self):
        """Process audio chunks and make predictions"""
        audio_buffer = np.array([])

        while self.is_running:
            try:
                # Get audio data from queue
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer = np.append(audio_buffer, chunk.flatten())

                # Process when we have enough samples
                if len(audio_buffer) >= self.chunk_samples:
                    # Take chunk
                    audio_chunk = audio_buffer[:self.chunk_samples]
                    audio_buffer = audio_buffer[self.chunk_samples:]

                    # Extract features and classify
                    try:
                        features = self.feature_extractor.extract_features(audio_chunk)
                        feature_vector = np.array([list(features.values())])
                        feature_vector_scaled = self.scaler.transform(feature_vector)

                        # Predict
                        prediction = self.trained_model.predict(feature_vector_scaled)[0]
                        probabilities = self.trained_model.predict_proba(feature_vector_scaled)[0]

                        # Store results
                        self.current_prediction = prediction
                        self.current_probabilities = probabilities
                        self.current_features = features  # Store raw features

                        self.predictions.append(prediction)
                        self.confidences.append(max(probabilities))
                        self.probabilities_history.append(probabilities)
                        self.timestamps.append(self.time_counter)
                        self.time_counter += self.chunk_duration

                    except Exception as e:
                        print(f"Error processing audio: {e}")

            except queue.Empty:
                continue

    def start_recording(self):
        """Start audio recording and processing"""
        self.is_running = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()

        # Start audio stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sr,
            callback=self.audio_callback,
            blocksize=int(self.sr * 0.1)  # 100ms blocks
        )
        self.stream.start()

        print("Recording started! Speak into your microphone...")

    def stop_recording(self):
        """Stop audio recording and processing"""
        self.is_running = False

        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()

        print("Recording stopped.")

    def visualize(self):
        """Create real-time visualization"""
        fig = plt.figure(figsize=(16, 10))

        # Create subplots
        if self.feature_importance is not None:
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, :])  # Current prediction
            ax2 = fig.add_subplot(gs[1, :])  # Prediction history
            ax3 = fig.add_subplot(gs[2, 0])  # Probability distribution
            ax4 = fig.add_subplot(gs[2, 1])  # Confidence over time
            ax5 = fig.add_subplot(gs[2, 2])  # Top 5 features
        else:
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, :])  # Current prediction
            ax2 = fig.add_subplot(gs[1, :])  # Prediction history
            ax3 = fig.add_subplot(gs[2, 0])  # Probability distribution
            ax4 = fig.add_subplot(gs[2, 1])  # Confidence over time
            ax5 = None

        # Color mapping for classes
        colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71'][:self.n_classes]

        def init():
            """Initialize plots"""
            return []

        def update(frame):
            """Update plots"""
            # Clear all axes
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            if ax5 is not None:
                ax5.clear()

            if self.current_prediction is None:
                ax1.text(0.5, 0.5, 'Waiting for audio...',
                        ha='center', va='center', fontsize=20)
                ax1.axis('off')
                return []

            # 1. Current prediction (large display)
            pred_class = self.class_names[self.current_prediction]
            confidence = max(self.current_probabilities) * 100

            ax1.text(0.5, 0.6, pred_class,
                    ha='center', va='center', fontsize=36,
                    fontweight='bold',
                    color=colors[self.current_prediction])
            ax1.text(0.5, 0.3, f'Confidence: {confidence:.1f}%',
                    ha='center', va='center', fontsize=20,
                    color='gray')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            ax1.set_title('Current Prediction', fontsize=16, fontweight='bold')

            # 2. Prediction history (timeline)
            if len(self.predictions) > 0:
                pred_array = np.array(self.predictions)
                time_array = np.array(self.timestamps)

                # Plot as colored segments
                for i in range(len(pred_array)):
                    ax2.barh(0, self.chunk_duration,
                            left=time_array[i],
                            color=colors[pred_array[i]],
                            alpha=0.7,
                            edgecolor='white',
                            linewidth=1)

                ax2.set_xlim(max(0, self.time_counter - self.buffer_size * self.chunk_duration),
                            self.time_counter + self.chunk_duration)
                ax2.set_ylim(-0.5, 0.5)
                ax2.set_xlabel('Time (seconds)', fontsize=12)
                ax2.set_yticks([])
                ax2.set_title('Prediction History', fontsize=14, fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)

                # Add legend
                handles = [plt.Rectangle((0,0),1,1, color=colors[i])
                          for i in range(self.n_classes)]
                ax2.legend(handles, self.class_names,
                          loc='upper right', ncol=self.n_classes)

            # 3. Current probability distribution (bar chart)
            if self.current_probabilities is not None:
                bars = ax3.bar(range(self.n_classes),
                              self.current_probabilities * 100,
                              color=colors,
                              alpha=0.7,
                              edgecolor='black',
                              linewidth=1.5)

                # Add percentage labels on bars
                for i, (bar, prob) in enumerate(zip(bars, self.current_probabilities)):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                            f'{prob*100:.1f}%',
                            ha='center', va='bottom', fontsize=10)

                ax3.set_xticks(range(self.n_classes))
                ax3.set_xticklabels(self.class_names, rotation=15, ha='right')
                ax3.set_ylabel('Probability (%)', fontsize=12)
                ax3.set_ylim(0, 105)
                ax3.set_title('Current Class Probabilities', fontsize=14, fontweight='bold')
                ax3.grid(axis='y', alpha=0.3)

            # 4. Confidence over time (line plot)
            if len(self.confidences) > 0:
                conf_array = np.array(self.confidences) * 100
                time_array = np.array(self.timestamps)

                ax4.plot(time_array, conf_array,
                        color='#2ecc71', linewidth=2.5,
                        marker='o', markersize=5)
                ax4.fill_between(time_array, conf_array,
                                alpha=0.3, color='#2ecc71')

                # Add horizontal line at 80% confidence
                ax4.axhline(y=80, color='gray', linestyle='--',
                           alpha=0.5, label='80% threshold')

                ax4.set_xlim(max(0, self.time_counter - self.buffer_size * self.chunk_duration),
                            self.time_counter + self.chunk_duration)
                ax4.set_ylim(0, 105)
                ax4.set_xlabel('Time (seconds)', fontsize=12)
                ax4.set_ylabel('Confidence (%)', fontsize=12)
                ax4.set_title('Confidence Over Time', fontsize=14, fontweight='bold')
                ax4.grid(alpha=0.3)
                ax4.legend()

            # 5. Top 5 most important features (if available)
            if ax5 is not None and self.current_features is not None and self.feature_importance is not None:
                # Get top 5 features
                top_5_features = self.feature_importance[:5]

                # Get current values for these features
                feature_values = []
                feature_labels = []

                for feat_name, importance in top_5_features:
                    if feat_name in self.current_features:
                        value = self.current_features[feat_name]
                        feature_values.append(value)
                        # Shorten feature name for display
                        short_name = feat_name.replace('_mean', '').replace('_std', '').replace('spectral_', 's_')
                        feature_labels.append(f"{short_name[:15]}")

                if feature_values:
                    # Plot horizontal bars
                    y_pos = np.arange(len(feature_labels))
                    bars = ax5.barh(y_pos, feature_values, color='#9b59b6', alpha=0.7, edgecolor='black')

                    ax5.set_yticks(y_pos)
                    ax5.set_yticklabels(feature_labels, fontsize=9)
                    ax5.set_xlabel('Feature Value', fontsize=10)
                    ax5.set_title('Top 5 Features', fontsize=12, fontweight='bold')
                    ax5.grid(axis='x', alpha=0.3)

                    # Add value labels
                    for i, (bar, val) in enumerate(zip(bars, feature_values)):
                        width = bar.get_width()
                        ax5.text(width, bar.get_y() + bar.get_height()/2,
                                f'{val:.2f}',
                                ha='left', va='center', fontsize=8,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

            return []

        # Create animation
        ani = animation.FuncAnimation(
            fig, update, init_func=init,
            interval=100,  # Update every 100ms
            blit=False
        )

        plt.suptitle('Real-time Microphone Distance Classification',
                    fontsize=18, fontweight='bold', y=0.98)

        return fig, ani


def main():
    """Main function to run real-time classifier"""
    import sys

    # Model path
    model_path = 'trained_model.pkl'

    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    try:
        # Create classifier
        print("Initializing real-time classifier...")
        rt_classifier = RealtimeAudioClassifier(
            model_path=model_path,
            sr=16000,
            chunk_duration=1.0,  # Analyze every 1 second
            buffer_size=20  # Keep last 20 predictions
        )

        # Start recording
        rt_classifier.start_recording()

        # Create visualization
        print("Starting visualization...")
        print("Press Ctrl+C or close the window to stop.")
        fig, ani = rt_classifier.visualize()

        # Show plot
        plt.show()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo use this script, you need to:")
        print("1. Train a model using AudioDistance.py")
        print("2. Save the model using the save_model() method")
        print("3. Run this script with the model path")
        sys.exit(1)

    except KeyboardInterrupt:
        print("\nStopping...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if 'rt_classifier' in locals():
            rt_classifier.stop_recording()
        print("Done!")


if __name__ == "__main__":
    main()
