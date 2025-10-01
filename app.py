import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import time
import threading
from collections import deque
import random

class HandDetectionGame:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Game variables
        self.score = 0
        self.game_time = 60  # seconds
        self.start_time = None
        self.is_running = False
        self.targets = []
        self.hand_data = []
        self.cap = None
        
        # Visualization data
        self.landmark_history = deque(maxlen=100)
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Hand Detection Game - AI Analytics")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Camera and Game
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Camera display
        self.camera_label = ttk.Label(left_frame, background='black')
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Game info
        info_frame = ttk.Frame(left_frame)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.score_label = ttk.Label(info_frame, text="Score: 0", font=('Arial', 14))
        self.score_label.pack(side=tk.LEFT, padx=10)
        
        self.time_label = ttk.Label(info_frame, text="Time: 60s", font=('Arial', 14))
        self.time_label.pack(side=tk.LEFT, padx=10)
        
        # Control buttons
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_btn = ttk.Button(control_frame, text="Start Game", command=self.start_game)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_game, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.visualize_btn = ttk.Button(control_frame, text="Show Analytics", command=self.show_analytics)
        self.visualize_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Analytics
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Real-time data tab
        self.realtime_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.realtime_tab, text="Real-time Data")
        
        # Analytics tab
        self.analytics_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_tab, text="Advanced Analytics")
        
        # Setup real-time plots
        self.setup_realtime_plots()
        
    def setup_realtime_plots(self):
        # Create matplotlib figures for real-time data
        self.realtime_fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        self.realtime_fig.tight_layout(pad=3.0)
        
        # Embed in tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.realtime_canvas = FigureCanvasTkAgg(self.realtime_fig, master=self.realtime_tab)
        self.realtime_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def start_game(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open camera")
                return
        
        self.score = 0
        self.game_time = 60
        self.start_time = time.time()
        self.is_running = True
        self.targets = []
        self.hand_data = []
        self.landmark_history.clear()
        
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_score_display()
        
        # Start game loop in separate thread
        self.game_thread = threading.Thread(target=self.game_loop)
        self.game_thread.daemon = True
        self.game_thread.start()
        
        # Start countdown
        self.update_timer()
        
    def stop_game(self):
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
    def update_timer(self):
        if self.is_running:
            elapsed = time.time() - self.start_time
            remaining = max(0, self.game_time - int(elapsed))
            self.time_label.config(text=f"Time: {remaining}s")
            
            if remaining <= 0:
                self.stop_game()
                messagebox.showinfo("Game Over", f"Game Over! Final Score: {self.score}")
            else:
                self.root.after(1000, self.update_timer)
    
    def update_score_display(self):
        self.score_label.config(text=f"Score: {self.score}")
    
    def generate_target(self, frame_shape):
        """Generate a new target circle"""
        radius = 30
        x = random.randint(radius, frame_shape[1] - radius)
        y = random.randint(radius, frame_shape[0] - radius)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        return {'x': x, 'y': y, 'radius': radius, 'color': color}
    
    def check_target_collision(self, hand_landmarks, frame_shape):
        """Check if hand landmarks collide with any targets"""
        if not hand_landmarks:
            return False
        
        for target in self.targets[:]:
            # Use index finger tip (landmark 8)
            landmark = hand_landmarks.landmark[8]
            x = int(landmark.x * frame_shape[1])
            y = int(landmark.y * frame_shape[0])
            
            distance = np.sqrt((x - target['x'])**2 + (y - target['y'])**2)
            
            if distance < target['radius']:
                self.targets.remove(target)
                self.score += 1
                self.update_score_display()
                return True
        return False
    
    def process_hand_data(self, hand_landmarks):
        """Extract and store hand landmark data for analytics"""
        if hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            self.landmark_history.append(landmarks)
            self.hand_data.append({
                'timestamp': time.time(),
                'landmarks': landmarks,
                'score': self.score
            })
    
    def update_realtime_plots(self):
        """Update real-time visualization plots"""
        if len(self.landmark_history) < 2:
            return
            
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Convert to numpy array
        data = np.array(list(self.landmark_history))
        
        # Plot 1: Hand landmark movement over time
        if len(data) > 10:
            self.ax1.plot(data[-50:, 0], data[-50:, 1], 'b-', alpha=0.7)
            self.ax1.scatter(data[-1, 0], data[-1, 1], c='red', s=50)
            self.ax1.set_title('Hand Movement Pattern')
            self.ax1.set_xlabel('X Coordinate')
            self.ax1.set_ylabel('Y Coordinate')
            self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Score progression
        if len(self.hand_data) > 5:
            scores = [data['score'] for data in self.hand_data[-50:]]
            self.ax2.plot(range(len(scores)), scores, 'g-', linewidth=2)
            self.ax2.set_title('Score Progression')
            self.ax2.set_xlabel('Time Steps')
            self.ax2.set_ylabel('Score')
            self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: Landmark velocity
        if len(data) > 2:
            velocity = np.sqrt(np.sum(np.diff(data[-20:], axis=0)**2, axis=1))
            self.ax3.plot(velocity, 'orange', linewidth=2)
            self.ax3.set_title('Hand Movement Velocity')
            self.ax3.set_xlabel('Time Steps')
            self.ax3.set_ylabel('Velocity')
            self.ax3.grid(True, alpha=0.3)
        
        # Plot 4: 3D hand position distribution
        if len(data) > 10:
            self.ax4.scatter(data[-50:, 0], data[-50:, 1], c=data[-50:, 2], 
                            cmap='viridis', alpha=0.6)
            self.ax4.set_title('3D Position Distribution')
            self.ax4.set_xlabel('X Coordinate')
            self.ax4.set_ylabel('Y Coordinate')
        
        self.realtime_canvas.draw()
    
    def show_analytics(self):
        """Show advanced analytics in a new window"""
        if len(self.hand_data) < 10:
            messagebox.showwarning("Warning", "Not enough data for analytics. Play the game first!")
            return
        
        analytics_window = tk.Toplevel(self.root)
        analytics_window.title("Advanced Hand Detection Analytics")
        analytics_window.geometry("1200x800")
        
        # Create notebook for analytics tabs
        analytics_notebook = ttk.Notebook(analytics_window)
        analytics_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Prepare data for ML algorithms
        landmarks_data = []
        scores = []
        
        for data_point in self.hand_data:
            landmarks_data.append(data_point['landmarks'])
            scores.append(data_point['score'])
        
        X = np.array(landmarks_data)
        y = np.array(scores)
        
        # Tab 1: t-SNE Visualization
        tsne_tab = ttk.Frame(analytics_notebook)
        analytics_notebook.add(tsne_tab, text="t-SNE Clustering")
        
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_tsne = tsne.fit_transform(X)
        
        scatter = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
        ax1.set_title('t-SNE Visualization of Hand Landmarks')
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        plt.colorbar(scatter, ax=ax1, label='Score')
        
        canvas1 = FigureCanvasTkAgg(fig1, master=tsne_tab)
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas1.draw()
        
        # Tab 2: PCA Analysis
        pca_tab = ttk.Frame(analytics_notebook)
        analytics_notebook.add(pca_tab, text="PCA Analysis")
        
        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.7)
        ax2.set_title('PCA of Hand Landmarks')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # Explained variance ratio
        pca_full = PCA().fit(X)
        ax3.plot(range(1, len(pca_full.explained_variance_ratio_) + 1), 
                np.cumsum(pca_full.explained_variance_ratio_), 'bo-')
        ax3.set_title('Cumulative Explained Variance')
        ax3.set_xlabel('Number of Components')
        ax3.set_ylabel('Cumulative Explained Variance')
        ax3.grid(True, alpha=0.3)
        
        canvas2 = FigureCanvasTkAgg(fig2, master=pca_tab)
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas2.draw()
        
        # Tab 3: K-means Clustering
        kmeans_tab = ttk.Frame(analytics_notebook)
        analytics_notebook.add(kmeans_tab, text="K-means Clustering")
        
        fig3, ax4 = plt.subplots(figsize=(10, 8))
        
        # Apply K-means
        kmeans = KMeans(n_clusters=min(5, len(X)), random_state=42)
        clusters = kmeans.fit_predict(X)
        
        scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
        ax4.set_title('K-means Clustering of Hand Positions')
        ax4.set_xlabel('PC1')
        ax4.set_ylabel('PC2')
        
        canvas3 = FigureCanvasTkAgg(fig3, master=kmeans_tab)
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas3.draw()
        
        # Tab 4: Statistical Analysis
        stats_tab = ttk.Frame(analytics_notebook)
        analytics_notebook.add(stats_tab, text="Statistical Analysis")
        
        fig4, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Landmark position distribution
        landmark_means = np.mean(X, axis=0)
        ax5.bar(range(len(landmark_means[:21])), landmark_means[:21])
        ax5.set_title('Average Landmark Positions (First 21)')
        ax5.set_xlabel('Landmark Index')
        ax5.set_ylabel('Average Position')
        
        # Movement patterns
        movement_magnitude = np.std(X, axis=0)
        ax6.bar(range(len(movement_magnitude[:21])), movement_magnitude[:21])
        ax6.set_title('Landmark Movement Variability')
        ax6.set_xlabel('Landmark Index')
        ax6.set_ylabel('Standard Deviation')
        
        # Score distribution
        ax7.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax7.set_title('Score Distribution')
        ax7.set_xlabel('Score')
        ax7.set_ylabel('Frequency')
        
        # Correlation heatmap (first 10 landmarks)
        corr_matrix = np.corrcoef(X[:, :30].T)
        im = ax8.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        ax8.set_title('Landmark Correlation Matrix')
        plt.colorbar(im, ax=ax8)
        
        fig4.tight_layout(pad=3.0)
        canvas4 = FigureCanvasTkAgg(fig4, master=stats_tab)
        canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas4.draw()
    
    def game_loop(self):
        """Main game loop running in separate thread"""
        while self.is_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe Hands
            results = self.hands.process(rgb_frame)
            
            # Generate targets if needed
            if len(self.targets) < 3 and random.random() < 0.02:
                self.targets.append(self.generate_target(frame.shape))
            
            # Draw targets
            for target in self.targets:
                cv2.circle(frame, (target['x'], target['y']), 
                          target['radius'], target['color'], -1)
                cv2.circle(frame, (target['x'], target['y']), 
                          target['radius'], (255, 255, 255), 2)
            
            # Process hand detection
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Check for target collisions
                    self.check_target_collision(hand_landmarks, frame.shape)
                    
                    # Process hand data for analytics
                    self.process_hand_data(hand_landmarks)
            
            # Convert frame for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update camera display in main thread
            self.root.after(0, self.update_camera_display, imgtk)
            
            # Update real-time plots
            if len(self.landmark_history) > 0:
                self.root.after(0, self.update_realtime_plots)
            
            # Control frame rate
            time.sleep(0.03)
    
    def update_camera_display(self, imgtk):
        """Update camera display in main thread"""
        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)
    
    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        finally:
            if self.cap:
                self.cap.release()
            self.hands.close()

if __name__ == "__main__":
    # Check dependencies
    try:
        game = HandDetectionGame()
        print("Starting Hand Detection Game...")
        print("Features:")
        print("- Real-time hand detection using MediaPipe")
        print("- Interactive target collection game")
        print("- Real-time data visualization")
        print("- Advanced ML analytics (t-SNE, PCA, K-means)")
        print("- Statistical analysis of hand movements")
        game.run()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install opencv-python mediapipe matplotlib pandas scikit-learn seaborn pillow")
    except Exception as e:
        print(f"Error: {e}")