"""
Desktop GUI application for the chatbot.
Provides a user-friendly interface for interacting with the PC usage chatbot.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
import json
from training_data import simple_embed
from model_train import create_model
import os


class ChatbotGUI:
    """Main GUI application class for the chatbot."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PC Usage Chatbot")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Set window icon (if available)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
        
        # Load model
        self.model = None
        self.load_model()
        
        # Setup UI
        self.setup_ui()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_model(self):
        """Load the trained model."""
        try:
            if not os.path.exists('intent_model.pt'):
                messagebox.showerror(
                    "Error", 
                    "Model file 'intent_model.pt' not found.\nPlease train the model first using model_train.py"
                )
                self.root.destroy()
                return
            
            self.model = create_model()
            self.model.load_state_dict(torch.load('intent_model.pt'))
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.root.destroy()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Title label
        title_label = ttk.Label(
            main_frame, 
            text="PC Usage Chatbot", 
            font=("Arial", 18, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Chat display area (read-only)
        chat_frame = ttk.LabelFrame(main_frame, text="Conversation", padding="5")
        chat_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=70,
            height=25,
            font=("Arial", 11),
            state=tk.DISABLED,
            bg="#f5f5f5",
            relief=tk.SOLID,
            borderwidth=1
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # User input field
        self.user_input = ttk.Entry(
            input_frame,
            font=("Arial", 11),
            width=50
        )
        self.user_input.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.user_input.bind("<Return>", lambda e: self.send_message())
        
        # Send button
        send_button = ttk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            width=15
        )
        send_button.grid(row=0, column=1, sticky=tk.E)
        
        # Clear button
        clear_button = ttk.Button(
            input_frame,
            text="Clear Chat",
            command=self.clear_chat,
            width=15
        )
        clear_button.grid(row=0, column=2, sticky=tk.E, padx=(5, 0))
        
        # Status bar
        self.status_bar = ttk.Label(
            main_frame,
            text="Ready - Type your question and press Enter or click Send",
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding="5"
        )
        self.status_bar.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        # Welcome message
        self.add_message("Bot", "Hello! I'm your PC Usage Chatbot. Ask me about:\n"
                                "• Which app you used most\n"
                                "• Your usage time and statistics\n\n"
                                "Type 'exit' to close the application.")
        
        # Focus on input field
        self.user_input.focus()
    
    def add_message(self, sender, message):
        """Add a message to the chat display."""
        self.chat_display.config(state=tk.NORMAL)
        
        # Format message based on sender
        if sender == "You":
            self.chat_display.insert(tk.END, f"You: {message}\n", "user")
        else:
            self.chat_display.insert(tk.END, f"Bot: {message}\n", "bot")
        
        # Configure tags for styling
        self.chat_display.tag_config("user", foreground="#0066cc", font=("Arial", 11, "bold"))
        self.chat_display.tag_config("bot", foreground="#008800", font=("Arial", 11))
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def predict_intent(self, text):
        """Predict the intent of user query."""
        if self.model is None:
            return None
        try:
            x = torch.tensor([simple_embed(text)], dtype=torch.float32)
            intent = torch.argmax(self.model(x)).item()
            return intent
        except Exception as e:
            print(f"Error predicting intent: {e}")
            return None
    
    def respond_to_query(self, text):
        """Generate response based on user query."""
        try:
            # Load usage data
            if not os.path.exists('usage_data.json'):
                return "Error: usage_data.json file not found. Please run tracker.py first."
            
            with open('usage_data.json', 'r') as f:
                usage = json.load(f)
            
            if not usage:
                return "No usage data available. Please run tracker.py to collect usage data."
            
            # Predict intent
            intent = self.predict_intent(text)
            
            if intent is None:
                return "Sorry, I encountered an error processing your request."
            
            # Generate response based on intent
            if intent == 0:  # most_used
                most_used = max(usage, key=usage.get)
                return f"You used '{most_used}' the most this week."
            else:  # usage_summary
                total = sum(usage.values())
                minutes = total / 60
                hours = minutes / 60
                num_apps = len(usage)
                return f"Total active time: {hours:.2f} hours ({minutes:.1f} minutes) across {num_apps} apps this week."
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def send_message(self):
        """Handle sending a message."""
        query = self.user_input.get().strip()
        
        if not query:
            return
        
        # Check for exit command
        if query.lower() in ['exit', 'quit', 'close']:
            self.on_closing()
            return
        
        # Add user message to chat
        self.add_message("You", query)
        
        # Clear input field
        self.user_input.delete(0, tk.END)
        
        # Update status
        self.status_bar.config(text="Processing...")
        self.root.update()
        
        # Get bot response
        response = self.respond_to_query(query)
        self.add_message("Bot", response)
        
        # Update status
        self.status_bar.config(text="Ready - Type your question and press Enter or click Send")
        
        # Focus back on input
        self.user_input.focus()
    
    def clear_chat(self):
        """Clear the chat display."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.add_message("Bot", "Chat cleared. How can I help you?")
    
    def on_closing(self):
        """Handle window closing."""
        if messagebox.askokcancel("Quit", "Do you want to exit the chatbot?"):
            self.root.destroy()


def main():
    """Main function to start the GUI application."""
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

