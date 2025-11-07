import numpy as np
import librosa
import pandas as pd
from scipy import signal, stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """Preprocessing audio souborů - rozdělení na chunky"""
    
    def __init__(self, chunk_duration=3.0, overlap=0.5, sr=16000, 
                 min_silence_duration=0.1, silence_threshold=-40):
        """
        Args:
            chunk_duration: délka jednoho chunku v sekundách
            overlap: překryv mezi chunky (0.0 = žádný, 0.5 = 50% překryv)
            sr: sample rate
            min_silence_duration: minimální délka ticha v sekundách pro detekci
            silence_threshold: práh pro detekci ticha v dB
        """
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.sr = sr
        self.min_silence_duration = min_silence_duration
        self.silence_threshold = silence_threshold
    
    def split_into_chunks(self, audio_path, remove_silence=True):
        """
        Rozdělí audio soubor na chunky
        
        Args:
            audio_path: cesta k audio souboru
            remove_silence: zda odstraňovat tiché části
            
        Returns:
            list: seznam audio chunků (numpy arrays)
            list: seznam časových pozic začátku každého chunku
        """
        # Načtení audia
        y, sr = librosa.load(audio_path, sr=self.sr)
        
        # Odstranění ticha pokud je požadováno
        if remove_silence:
            y = self._remove_silence(y, sr)
        
        # Pokud je audio příliš krátké, vrátíme ho celé
        if len(y) < self.chunk_duration * sr:
            return [y], [0.0]
        
        # Výpočet parametrů pro chunking
        chunk_samples = int(self.chunk_duration * sr)
        hop_samples = int(chunk_samples * (1 - self.overlap))
        
        chunks = []
        timestamps = []
        
        # Rozdělení na chunky
        for start in range(0, len(y) - chunk_samples + 1, hop_samples):
            end = start + chunk_samples
            chunk = y[start:end]
            
            # Kontrola, zda chunk není příliš tichý
            if self._is_valid_chunk(chunk):
                chunks.append(chunk)
                timestamps.append(start / sr)
        
        # Pokud máme zbytek, přidáme ho jako poslední chunk (s paddingem)
        if len(chunks) > 0:  # Pouze pokud už máme nějaké chunky
            remainder = len(y) % hop_samples
            if remainder > sr * 0.5:  # Pokud je zbytek delší než 0.5s
                last_chunk = y[-chunk_samples:]
                if len(last_chunk) < chunk_samples:
                    # Padding
                    last_chunk = np.pad(last_chunk, (0, chunk_samples - len(last_chunk)))
                if self._is_valid_chunk(last_chunk):
                    chunks.append(last_chunk)
                    timestamps.append((len(y) - chunk_samples) / sr)
        
        return chunks, timestamps
    
    def _remove_silence(self, y, sr):
        """Odstraní tiché části z audia"""
        # Výpočet RMS energie
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Převod na dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Najdeme frame, které nejsou tiché
        non_silent_frames = rms_db > self.silence_threshold
        
        # Převod framů na sample indexy
        non_silent_intervals = librosa.frames_to_samples(
            np.where(non_silent_frames)[0], 
            hop_length=hop_length
        )
        
        if len(non_silent_intervals) == 0:
            return y  # Pokud je vše ticho, vrátíme originál
        
        # Spojíme blízké intervaly
        min_silence_samples = int(self.min_silence_duration * sr)
        
        # Extrakce non-silent částí
        result = []
        current_start = non_silent_intervals[0]
        current_end = non_silent_intervals[0]
        
        for idx in non_silent_intervals[1:]:
            if idx - current_end < min_silence_samples:
                current_end = idx
            else:
                result.append(y[current_start:current_end])
                current_start = idx
                current_end = idx
        
        # Přidáme poslední segment
        result.append(y[current_start:min(current_end + hop_length, len(y))])
        
        # Spojíme všechny non-silent části
        return np.concatenate(result)
    
    def _is_valid_chunk(self, chunk):
        """Zkontroluje, zda je chunk validní (není příliš tichý)"""
        rms = np.sqrt(np.mean(chunk**2))
        rms_db = 20 * np.log10(rms + 1e-10)
        return rms_db > self.silence_threshold
    
    def preprocess_dataset(self, audio_files, labels, augment=False):
        """
        Preprocessuje celý dataset - rozdělí všechny soubory na chunky
        
        Args:
            audio_files: seznam cest k audio souborům
            labels: seznam labelů
            augment: zda aplikovat data augmentaci
            
        Returns:
            chunk_files: seznam informací o chuncích
            chunk_labels: labely pro chunky
        """
        chunk_data = []
        chunk_labels = []
        
        print("Preprocessing audio files...")
        for idx, (audio_file, label) in enumerate(zip(audio_files, labels)):
            try:
                # Rozdělení na chunky
                chunks, timestamps = self.split_into_chunks(audio_file)
                
                print(f"[{idx+1}/{len(audio_files)}] {Path(audio_file).name}: {len(chunks)} chunks")
                
                for chunk_idx, (chunk, timestamp) in enumerate(zip(chunks, timestamps)):
                    chunk_info = {
                        'original_file': audio_file,
                        'chunk_audio': chunk,
                        'chunk_index': chunk_idx,
                        'timestamp': timestamp,
                        'augmented': False
                    }
                    chunk_data.append(chunk_info)
                    chunk_labels.append(label)
                    
                    # Data augmentace pokud je požadována
                    if augment and len(chunks) > 0:
                        aug_chunks = self._augment_chunk(chunk)
                        for aug_idx, aug_chunk in enumerate(aug_chunks):
                            aug_info = {
                                'original_file': audio_file,
                                'chunk_audio': aug_chunk,
                                'chunk_index': f"{chunk_idx}_aug_{aug_idx}",
                                'timestamp': timestamp,
                                'augmented': True
                            }
                            chunk_data.append(aug_info)
                            chunk_labels.append(label)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        print(f"\nTotal chunks created: {len(chunk_data)}")
        print(f"Original chunks: {sum(1 for c in chunk_data if not c['augmented'])}")
        print(f"Augmented chunks: {sum(1 for c in chunk_data if c['augmented'])}")
        
        return chunk_data, chunk_labels
    
    def _augment_chunk(self, chunk):
        """
        Aplikuje data augmentaci na chunk
        
        Returns:
            list: seznam augmentovaných chunků
        """
        augmented = []
        
        # 1. Time stretching (zrychlení/zpomalení)
        try:
            stretched = librosa.effects.time_stretch(chunk, rate=1.1)
            if len(stretched) >= len(chunk):
                augmented.append(stretched[:len(chunk)])
            else:
                augmented.append(np.pad(stretched, (0, len(chunk) - len(stretched))))
        except:
            pass
        
        # 2. Pitch shifting (změna výšky tónu)
        try:
            pitched = librosa.effects.pitch_shift(chunk, sr=self.sr, n_steps=2)
            augmented.append(pitched)
        except:
            pass
        
        # 3. Přidání šumu
        noise = np.random.normal(0, 0.005, len(chunk))
        noisy = chunk + noise
        augmented.append(noisy)
        
        # 4. Změna hlasitosti
        louder = chunk * 1.2
        louder = np.clip(louder, -1.0, 1.0)
        augmented.append(louder)
        
        quieter = chunk * 0.8
        augmented.append(quieter)
        
        return augmented


class AudioFeatureExtractor:
    """Extrakce features pro detekci vzdálenosti od mikrofonu"""
    
    def __init__(self, sr=16000, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc
        
    def extract_features(self, audio_input):
        """
        Extrahuje všechny features z audio souboru nebo audio dat
        
        Args:
            audio_input: cesta k audio souboru NEBO numpy array s audio daty
            
        Returns:
            dict: slovník s features
        """
        # Načtení audia
        if isinstance(audio_input, str):
            y, sr = librosa.load(audio_input, sr=self.sr)
        else:
            y = audio_input
            sr = self.sr
        
        features = {}
        
        # 1. ENERGETICKÉ FEATURES
        features.update(self._extract_energy_features(y))
        
        # 2. SPEKTRÁLNÍ FEATURES
        features.update(self._extract_spectral_features(y, sr))
        
        # 3. CLARITY FEATURES
        features.update(self._extract_clarity_features(y, sr))
        
        # 4. ROOM ACOUSTICS FEATURES
        features.update(self._extract_room_features(y, sr))
        
        # 5. MFCC FEATURES (pro dodatečnou informaci)
        features.update(self._extract_mfcc_features(y, sr))
        
        return features
    
    def _extract_energy_features(self, y):
        """Extrakce energetických features"""
        features = {}
        
        # RMS energy
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_max'] = np.max(rms)
        features['rms_min'] = np.min(rms)
        
        # Peak amplitude
        features['peak_amplitude'] = np.max(np.abs(y))
        
        # Dynamic range
        features['dynamic_range'] = 20 * np.log10(np.max(np.abs(y)) / (np.min(np.abs(y)) + 1e-10))
        
        # Energy percentiles
        features['energy_25_percentile'] = np.percentile(rms, 25)
        features['energy_75_percentile'] = np.percentile(rms, 75)
        features['energy_iqr'] = features['energy_75_percentile'] - features['energy_25_percentile']
        
        return features
    
    def load_dataset_from_directories(self, base_path, class_mapping=None):
        """
        Načte dataset z adresářové struktury
        
        Args:
            base_path: kořenová cesta k datasetu
            class_mapping: dict mapující názvy složek na labely
                        např. {'very_far': 0, 'far': 1, 'close': 2, 'very_close': 3}
                        Pokud None, použije automatické mapování podle abecedy
            
        Returns:
            audio_files: seznam cest k audio souborům
            labels: seznam labelů
            class_names: seznam názvů tříd
            
        Očekávaná struktura:
            base_path/
                class_1/
                    audio1.wav
                    audio2.wav
                class_2/
                    audio1.wav
                    audio2.wav
        """
        from pathlib import Path
        
        base_path = Path(base_path)
        audio_files = []
        labels = []
        
        # Získání všech podadresářů
        subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
        
        # Vytvoření mapování, pokud nebylo poskytnuto
        if class_mapping is None:
            class_mapping = {d.name: idx for idx, d in enumerate(subdirs)}
            print(f"Automatické mapování tříd: {class_mapping}")
        
        # Podporované audio formáty
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aiff'}
        
        # Procházení adresářů
        for class_dir in subdirs:
            if class_dir.name not in class_mapping:
                print(f"Varování: Adresář '{class_dir.name}' není v class_mapping, přeskakuji")
                continue
                
            label = class_mapping[class_dir.name]
            
            # Hledání všech audio souborů v adresáři
            for audio_file in class_dir.iterdir():
                if audio_file.suffix.lower() in audio_extensions:
                    audio_files.append(str(audio_file))
                    labels.append(label)
            
            print(f"Načteno {len([l for l in labels if l == label])} souborů z třídy '{class_dir.name}' (label {label})")
        
        # Vytvoření seznamu názvů tříd seřazených podle labelů
        class_names = [None] * len(class_mapping)
        for name, idx in class_mapping.items():
            class_names[idx] = name
        
        print(f"\nCelkem načteno {len(audio_files)} audio souborů")
        print(f"Třídy: {class_names}")
        
        return audio_files, labels, class_names
    
    def _extract_spectral_features(self, y, sr):
        """Extrakce spektrálních features"""
        features = {}
        
        # Výpočet spektrogramu
        D = np.abs(librosa.stft(y))
        
        # Spectral centroid (těžiště spektra)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(cent)
        features['spectral_centroid_std'] = np.std(cent)
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        features['spectral_rolloff_mean'] = np.mean(rolloff)
        features['spectral_rolloff_std'] = np.std(rolloff)
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(bandwidth)
        features['spectral_bandwidth_std'] = np.std(bandwidth)
        
        # High frequency energy ratio (energie nad 4kHz)
        fft_freqs = librosa.fft_frequencies(sr=sr)
        high_freq_idx = np.where(fft_freqs >= 4000)[0]
        low_freq_idx = np.where(fft_freqs < 4000)[0]
        
        high_freq_energy = np.mean(np.sum(D[high_freq_idx, :], axis=0))
        low_freq_energy = np.mean(np.sum(D[low_freq_idx, :], axis=0))
        features['high_freq_ratio'] = high_freq_energy / (low_freq_energy + 1e-10)
        
        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(contrast.shape[0]):
            features[f'spectral_contrast_band_{i}'] = np.mean(contrast[i])
        
        return features
    
    def _extract_clarity_features(self, y, sr):
        """Extrakce features související s jasností zvuku"""
        features = {}
        
        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Spectral flatness (míra "šumovosti")
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = np.mean(flatness)
        features['spectral_flatness_std'] = np.std(flatness)
        
        # Estimated SNR (pomocí tichých částí)
        rms = librosa.feature.rms(y=y)[0]
        noise_floor = np.percentile(rms, 10)  # Nejnižších 10% jako šum
        signal_peak = np.percentile(rms, 90)  # Nejvyšších 10% jako signál
        features['estimated_snr'] = 20 * np.log10(signal_peak / (noise_floor + 1e-10))
        
        return features
    
    def _extract_room_features(self, y, sr):
        """Extrakce features související s prostorovou akustikou"""
        features = {}
        
        # Envelope decay rate (aproximace reverbu)
        envelope = np.abs(signal.hilbert(y))
        
        # Najdeme peaky
        peaks, _ = signal.find_peaks(envelope, height=np.max(envelope)*0.3)
        
        if len(peaks) > 1:
            # Měříme decay mezi peaky
            decay_rates = []
            for i in range(len(peaks)-1):
                segment = envelope[peaks[i]:peaks[i+1]]
                if len(segment) > 10:
                    # Fitujeme exponenciální decay
                    x = np.arange(len(segment))
                    log_segment = np.log(segment + 1e-10)
                    if not np.any(np.isnan(log_segment)):
                        slope, _ = np.polyfit(x, log_segment, 1)
                        decay_rates.append(slope)
            
            if decay_rates:
                features['envelope_decay_mean'] = np.mean(decay_rates)
                features['envelope_decay_std'] = np.std(decay_rates) if len(decay_rates) > 1 else 0
            else:
                features['envelope_decay_mean'] = 0
                features['envelope_decay_std'] = 0
        else:
            features['envelope_decay_mean'] = 0
            features['envelope_decay_std'] = 0
        
        # Direct-to-reverberant ratio (aproximace)
        # První 50ms považujeme za přímý zvuk
        direct_samples = int(0.05 * sr)
        if len(y) > direct_samples * 2:
            direct_energy = np.sum(y[:direct_samples]**2)
            reverb_energy = np.sum(y[direct_samples:]**2)
            features['direct_to_reverb_ratio'] = 10 * np.log10(direct_energy / (reverb_energy + 1e-10))
        else:
            features['direct_to_reverb_ratio'] = 0
            
        return features
    
    def _extract_mfcc_features(self, y, sr):
        """Extrakce MFCC features"""
        features = {}
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        
        for i in range(self.n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            
        # Delta MFCC (první derivace)
        mfcc_delta = librosa.feature.delta(mfccs)
        for i in range(min(5, self.n_mfcc)):  # Jen prvních 5 delta
            features[f'mfcc_delta_{i}_mean'] = np.mean(mfcc_delta[i])
            
        return features


class AudioDistanceClassifier:
    def __init__(self, model_type='xgboost', n_classes=2):
        self.model_type = model_type
        self.n_classes = n_classes
        self.feature_extractor = AudioFeatureExtractor()
        self.preprocessor = AudioPreprocessor()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = None
        self.class_names = None
        
    def prepare_dataset(self, audio_files, labels, use_chunks=True, augment=False):
        """
        Připraví dataset z audio souborů
        
        Args:
            audio_files: seznam cest k audio souborům
            labels: seznam labelů
            use_chunks: zda rozdělit soubory na chunky
            augment: zda použít data augmentaci
            
        Returns:
            X: feature matrix
            y: labels
            feature_names: názvy features
        """
        if use_chunks:
            # Preprocessing - rozdělení na chunky
            chunk_data, chunk_labels = self.preprocessor.preprocess_dataset(
                audio_files, labels, augment=augment
            )
            
            # Extrakce features z chunků
            features_list = []
            valid_labels = []
            
            print("\nExtracting features from chunks...")
            for idx, (chunk_info, label) in enumerate(zip(chunk_data, chunk_labels)):
                try:
                    # Extrakce features přímo z audio dat (ne ze souboru)
                    features = self.feature_extractor.extract_features(chunk_info['chunk_audio'])
                    features_list.append(features)
                    valid_labels.append(label)
                    
                    if (idx + 1) % 100 == 0:
                        print(f"Processed {idx + 1}/{len(chunk_data)} chunks")
                        
                except Exception as e:
                    print(f"Error extracting features from chunk {idx}: {e}")
                    continue
        else:
            # Bez chunkingu - použijeme celé soubory
            features_list = []
            valid_labels = []
            
            print("\nExtracting features from full audio files...")
            for audio_file, label in zip(audio_files, labels):
                try:
                    features = self.feature_extractor.extract_features(audio_file)
                    features_list.append(features)
                    valid_labels.append(label)
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
        
        # Převod na DataFrame
        df = pd.DataFrame(features_list)
        X = df.values
        y = np.array(valid_labels)
        
        print(f"\nFinal dataset size: {X.shape[0]} samples with {X.shape[1]} features")
        
        return X, y, df.columns
    
    def train(self, X, y, feature_names):
        """
        Trénuje model
        
        Args:
            X: feature matrix
            y: labels
            feature_names: názvy features
        """
        # Rozdělení dat
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Škálování features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Trénování modelu
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='multi:softmax',
                num_class=self.n_classes,
                random_state=42
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluace
        y_pred = self.model.predict(X_test_scaled)
        
        target_names = [
            'distance_ok',
            'distance_too_far',
        ]
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Feature importance
        importance = self.model.feature_importances_
            
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return self.model, self.feature_importance
    
    def plot_feature_importance(self, top_n=20):
        """Vizualizace důležitosti features"""
        print("plot_feature_importance")
        if self.feature_importance is None:
            print("Model ještě nebyl natrénován!")
            return

        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)

        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title(f'Top {top_n} nejdůležitějších features')
        plt.xlabel('Důležitost')
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath='trained_model.pkl'):
        """
        Save the trained model to a file

        Args:
            filepath: path where to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")

        # Prepare feature importance data
        feature_importance_list = None
        if self.feature_importance is not None:
            # Convert to list of tuples for easy iteration
            feature_importance_list = list(zip(
                self.feature_importance['feature'].values,
                self.feature_importance['importance'].values
            ))

        model_data = {
            'trained_model': self.model,
            'scaler': self.scaler,
            'class_names': self.class_names,
            'model_type': self.model_type,
            'n_classes': self.n_classes,
            'feature_extractor_params': {
                'sr': self.feature_extractor.sr,
                'n_mfcc': self.feature_extractor.n_mfcc
            },
            'feature_importance': feature_importance_list,
            'feature_names': self.feature_importance['feature'].values.tolist() if self.feature_importance is not None else None
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")
        if feature_importance_list:
            print(f"Saved with {len(feature_importance_list)} feature importances")

    @classmethod
    def load_model(cls, filepath='trained_model.pkl'):
        """
        Load a trained model from a file

        Args:
            filepath: path to the saved model

        Returns:
            AudioDistanceClassifier: loaded classifier
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create new classifier instance
        classifier = cls(
            model_type=model_data['model_type'],
            n_classes=model_data['n_classes']
        )

        # Restore trained components
        classifier.model = model_data['trained_model']
        classifier.scaler = model_data['scaler']
        classifier.class_names = model_data['class_names']

        # Restore feature extractor params
        fe_params = model_data['feature_extractor_params']
        classifier.feature_extractor = AudioFeatureExtractor(
            sr=fe_params['sr'],
            n_mfcc=fe_params['n_mfcc']
        )

        print(f"Model loaded from {filepath}")
        print(f"Classes: {classifier.class_names}")

        return classifier

    def get_interpretable_feedback(self, audio_path):
        """
        Vrátí interpretovatelnou zpětnou vazbu pro dabéra
        
        Args:
            audio_path: cesta k audio souboru
            
        Returns:
            dict: zpětná vazba s doporučeními
        """
        features = self.feature_extractor.extract_features(audio_path)
        feature_vector = np.array([list(features.values())])
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predikce
        prediction = self.model.predict(feature_vector_scaled)[0]
        probability = self.model.predict_proba(feature_vector_scaled)[0]
        
        distance_labels = {
            0: 'Velmi daleko od mikrofonu',
            1: 'Daleko od mikrofonu', 
            2: 'Blízko mikrofonu',
            3: 'Velmi blízko mikrofonu'
        }
        
        feedback = {
            'prediction': distance_labels.get(prediction, 'Unknown'),
            'confidence': max(probability) * 100,
            'all_probabilities': {
                distance_labels.get(i, f'Class {i}'): prob * 100 
                for i, prob in enumerate(probability)
            },
            'recommendations': []
        }
        
        # Adjust recommendations based on multi-class prediction
        if prediction == 0:  # Very far
            feedback['recommendations'].append("Significantly increase volume")
            feedback['recommendations'].append("Move much closer to microphone")
        elif prediction == 1:  # Far
            feedback['recommendations'].append("Increase volume moderately")
            feedback['recommendations'].append("Move closer to microphone")
        elif prediction == 2:  # Close
            feedback['recommendations'].append("Good distance, minor adjustments may help")
        else:  # Very close
            feedback['recommendations'].append("Excellent positioning!")
        
        return feedback

# Příklad použití
if __name__ == "__main__":
    # Initialize classifier
    classifier = AudioDistanceClassifier(model_type='xgboost', n_classes=2)
    extractor = AudioFeatureExtractor()
    
    # Define your distance categories
    # class_mapping = {
    #     'very_far': 0,    # > 2 meters
    #     'far': 1,         # 1-2 meters
    #     'close': 2,       # 0.5-1 meter
    #     'very_close': 3   # < 0.5 meter
    # }
    
    # Load dataset
    audio_files, labels, class_names = extractor.load_dataset_from_directories(
        'dataset'
    )
    
    # Store class names in classifier
    classifier.class_names = class_names

    # Prepare features
    X, y, feature_names = classifier.prepare_dataset(audio_files, labels)

    # Train model
    classifier.train(X, y, feature_names)

    # Visualize feature importance
    classifier.plot_feature_importance(top_n=20)

    # Save the trained model
    classifier.save_model('trained_model.pkl')

    # Test on new audio (if available)
    # feedback = classifier.get_interpretable_feedback('test_audio.wav')
    # print(f"\nPredikce: {feedback['prediction']}")
    # print(f"Jistota: {feedback['confidence']:.1f}%")
    # print("\nDoporučení:")
    # for rec in feedback['recommendations']:
    #     print(f"  • {rec}")