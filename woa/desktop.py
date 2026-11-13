import threading

from .client import request_json


class DesktopClient:
    def __init__(self):
        import tkinter as tk
        from tkinter import scrolledtext

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Woa")
        self.root.geometry("900x650")
        self.root.minsize(720, 500)
        self.active_id = None
        self.message_signature = None
        self.conversation_ids = []

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        sidebar = tk.Frame(self.root, padx=10, pady=10)
        sidebar.grid(row=0, column=0, sticky="ns")
        tk.Label(sidebar, text="Woa", font=("", 14, "bold")).pack(
            anchor="w", pady=(0, 10)
        )
        tk.Button(sidebar, text="＋ New conversation", command=self.new_conversation).pack(
            fill="x", pady=(0, 8)
        )
        self.conversations = tk.Listbox(sidebar, width=28, exportselection=False)
        self.conversations.pack(fill="both", expand=True)
        self.conversations.bind("<<ListboxSelect>>", self.on_select)

        main = tk.Frame(self.root, padx=12, pady=12)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)
        self.model_status = tk.Label(main, anchor="w", fg="#536581")
        self.model_status.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.training_status = tk.Label(main, anchor="w", fg="#536581")
        self.training_status.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.messages = scrolledtext.ScrolledText(
            main, wrap="word", state="disabled", padx=14, pady=14
        )
        self.messages.tag_configure(
            "user_label", foreground="#5478ae", font=("TkDefaultFont", 10, "bold")
        )
        self.messages.tag_configure(
            "assistant_label",
            foreground="#526f9d",
            font=("TkDefaultFont", 10, "bold"),
        )
        self.messages.tag_configure(
            "user_content", lmargin1=12, lmargin2=12, rmargin=24, spacing3=12
        )
        self.messages.tag_configure(
            "assistant_content", lmargin1=12, lmargin2=12, rmargin=12, spacing3=12
        )
        self.messages.tag_configure(
            "empty_state", foreground="#68758a", justify="center", spacing1=18
        )
        self.messages.grid(row=2, column=0, sticky="nsew")
        composer = tk.Frame(main)
        composer.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        composer.columnconfigure(0, weight=1)
        self.input = tk.Text(composer, height=3, wrap="word")
        self.input.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.input.bind("<Control-Return>", self.on_send_shortcut)
        self.send_button = tk.Button(composer, text="Send", command=self.send_message)
        self.send_button.grid(row=0, column=1, sticky="ns")
        tk.Label(
            composer,
            text="Enter for a new line · Ctrl+Enter to send",
            anchor="w",
            fg="#536581",
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.status = tk.Label(main, text="", anchor="w", fg="#a04d4d")
        self.status.grid(row=4, column=0, sticky="ew", pady=(6, 0))

    def run(self):
        self.refresh_health()
        self.refresh_conversations()
        self.refresh_training_status()
        self.root.after(1500, self.poll)
        self.root.mainloop()

    def refresh_health(self):
        try:
            health = request_json("/api/health")
            hardware = health["hardware"]
            accelerator = (
                f" ({hardware['accelerator']})" if hardware["accelerator"] else ""
            )
            if not health["model_available"]:
                text = health.get("model_message") or (
                    "No local model found; general chat needs a model."
                )
            elif health.get("model_backend") == "local_pretrained":
                text = f"Local pretrained chat model on {hardware['device']}{accelerator}"
            else:
                text = f"Experimental project model on {hardware['device']}{accelerator}"
            self.model_status.config(text=text)
        except RuntimeError as error:
            self.model_status.config(text=str(error))

    def refresh_conversations(self):
        try:
            conversations = request_json("/api/conversations")["conversations"]
        except RuntimeError as error:
            self.status.config(text=str(error))
            return
        selected_id = self.active_id
        self.conversation_ids = [item["id"] for item in conversations]
        self.conversations.delete(0, "end")
        for item in conversations:
            self.conversations.insert("end", item["title"])
        if selected_id in self.conversation_ids:
            index = self.conversation_ids.index(selected_id)
            self.conversations.selection_set(index)
        elif conversations:
            self.select_conversation(self.conversation_ids[0])
        elif not self.active_id:
            self.new_conversation()

    def poll(self):
        self.refresh_conversations()
        self.refresh_health()
        self.refresh_training_status()
        if self.active_id:
            self.refresh_messages()
        self.root.after(1500, self.poll)

    def refresh_training_status(self):
        try:
            events = request_json("/api/training/events")["events"]
        except RuntimeError as error:
            self.training_status.config(text=f"Training telemetry unavailable: {error}")
            return
        step = next(
            (event for event in events if event["event_type"] == "training_step"),
            None,
        )
        if step:
            payload = step["payload"]
            self.training_status.config(
                text=(
                    f"Training: step {payload['step']}/{payload['total_steps']}, "
                    f"loss {payload['loss']:.4f}, device {payload['device']}"
                )
            )
        elif events:
            self.training_status.config(
                text=f"Latest training event: {events[0]['event_type']}"
            )
        else:
            self.training_status.config(text="No local training telemetry yet.")

    def on_select(self, _event):
        selected = self.conversations.curselection()
        if selected:
            self.select_conversation(self.conversation_ids[selected[0]])

    def select_conversation(self, conversation_id):
        self.active_id = conversation_id
        self.message_signature = None
        self.refresh_messages()

    def new_conversation(self):
        try:
            result = request_json(
                "/api/conversations", {"title": "New conversation"}
            )
        except RuntimeError as error:
            self.status.config(text=str(error))
            return
        self.refresh_conversations()
        self.select_conversation(result["conversation"]["id"])

    def refresh_messages(self):
        conversation_id = self.active_id
        if not conversation_id:
            return
        try:
            result = request_json(
                f"/api/conversations/{conversation_id}/messages"
            )
        except RuntimeError as error:
            self.status.config(text=str(error))
            return
        if conversation_id != self.active_id:
            return
        messages = result["messages"]
        signature = ",".join(item["id"] for item in messages)
        if signature == self.message_signature:
            return
        self.message_signature = signature
        self.messages.config(state="normal")
        self.messages.delete("1.0", "end")
        if not messages:
            self.messages.insert(
                "end",
                "Start a conversation\nAsk a question, calculate, or search your local library.",
                "empty_state",
            )
        for item in messages:
            name = "You" if item["role"] == "user" else "Woa"
            role = "user" if item["role"] == "user" else "assistant"
            self.messages.insert("end", f"{name}\n", f"{role}_label")
            self.messages.insert(
                "end", f"{item['content']}\n\n", f"{role}_content"
            )
        self.messages.config(state="disabled")
        self.messages.see("end")

    def on_send_shortcut(self, _event):
        self.send_message()
        return "break"

    def send_message(self):
        content = self.input.get("1.0", "end").strip()
        if not content or not self.active_id:
            return
        self.input.delete("1.0", "end")
        self.send_button.config(state="disabled")
        self.status.config(text="Processing locally…")
        conversation_id = self.active_id

        def send():
            try:
                request_json(
                    f"/api/conversations/{conversation_id}/messages",
                    {"content": content},
                )
                self.root.after(0, self.refresh_messages)
                self.root.after(0, lambda: self.status.config(text=""))
            except RuntimeError as error:
                self.root.after(0, lambda message=str(error): self.status.config(text=message))
                def restore_input():
                    if conversation_id != self.active_id:
                        return
                    current = self.input.get("1.0", "end-1c")
                    self.input.insert("1.0", content + ("\n" if current else ""))
                    self.input.focus_set()

                self.root.after(0, restore_input)
            finally:
                self.root.after(0, lambda: self.send_button.config(state="normal"))

        threading.Thread(target=send, daemon=True).start()


def run():
    DesktopClient().run()
