const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("message-input");

const pdfInputEl = document.getElementById("pdf-input");
const uploadStatusEl = document.getElementById("upload-status");

// Botones de voz
const micBtn = document.getElementById("mic-btn");
const voiceBtn = document.getElementById("voice-btn");

let voiceEnabled = true;

// ====== TEXTO -> VOZ (TTS) ======
function cleanForTTS(text) {
  if (!text) return "";
  return text
    .replace(/\n+Fuentes:\s*[\s\S]*$/i, "")
    .replace(/\[[0-9]+\]/g, "")
    .trim();
}

function speak(text) {
  if (!voiceEnabled) return;
  if (!("speechSynthesis" in window)) return;

  const t = cleanForTTS(text);
  if (!t) return;

  window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(t);
  u.lang = "es-ES"; // puedes probar "es-EC"
  u.rate = 1;
  u.pitch = 1;
  window.speechSynthesis.speak(u);
}

voiceBtn.addEventListener("click", () => {
  voiceEnabled = !voiceEnabled;
  voiceBtn.classList.toggle("off", !voiceEnabled);
  voiceBtn.textContent = voiceEnabled ? "🔊" : "🔇";
});

// ====== VOZ -> TEXTO (STT) ======
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.lang = "es-ES";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  recognition.continuous = false;

  recognition.onstart = () => {
    isListening = true;
    micBtn.textContent = "🛑";
  };

  recognition.onend = () => {
    isListening = false;
    micBtn.textContent = "🎤";
  };

  recognition.onerror = (e) => {
    console.error("SpeechRecognition error:", e);
    addMessage("No pude usar el micrófono. Revisa permisos del navegador.", "bot", true);
  };

  recognition.onresult = (e) => {
    const transcript = (e.results?.[0]?.[0]?.transcript || "").trim();
    if (!transcript) return;

    inputEl.value = transcript;
    formEl.dispatchEvent(new Event("submit", { cancelable: true }));
  };

  micBtn.addEventListener("click", () => {
    try {
      if (isListening) recognition.stop();
      else recognition.start();
    } catch (err) {
      console.error(err);
    }
  });
} else {
  micBtn.disabled = true;
  micBtn.title = "Tu navegador no soporta SpeechRecognition";
  addMessage("Tu navegador no soporta dictado por voz. Prueba Chrome/Edge.", "bot", true);
}

function addMessage(text, sender = "bot", isSystem = false) {
  const row = document.createElement("div");
  row.classList.add("message-row", sender);

  const bubble = document.createElement("div");
  bubble.classList.add("message", sender);
  if (isSystem) bubble.classList.add("system");
  bubble.textContent = text;

  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  return row;
}

function addSources(row, sources) {
  if (!sources || !sources.length) return;

  const box = document.createElement("div");
  box.classList.add("sources-box");

  sources.forEach((s, idx) => {
    if (!s.file_name) return;

    const page = s.page || 1;

    const a = document.createElement("a");
    a.href = `/pdf/${encodeURIComponent(s.file_name)}#page=${page}`;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = `[${idx + 1}] ${s.file_name}, pág. ${page}`;

    box.appendChild(a);
  });

  row.appendChild(box);
}

// Mensaje inicial
addMessage(
  "Hola, soy el chatbot de LIRIS conectado a una base vectorial en Qdrant.\n" +
    "Puedes hacer preguntas y también adjuntar PDFs de la empresa para que queden indexados.",
  "bot"
);

// Manejo de chat
formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;

  addMessage(text, "user");
  inputEl.value = "";
  inputEl.focus();

  const typingRow = document.createElement("div");
  typingRow.classList.add("message-row", "bot");
  const typingBubble = document.createElement("div");
  typingBubble.classList.add("message", "bot", "typing");
  typingBubble.textContent = "Pensando...";
  typingRow.appendChild(typingBubble);
  messagesEl.appendChild(typingRow);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    const data = await res.json();
    messagesEl.removeChild(typingRow);

    if (!res.ok) {
      addMessage("Error: " + (data.error || "No se pudo procesar la respuesta"), "bot");
      return;
    }

    const botRow = addMessage(data.response, "bot");
    addSources(botRow, data.sources);

    // Habla la respuesta
    speak(data.response);
  } catch (err) {
    console.error(err);
    messagesEl.removeChild(typingRow);
    addMessage("Error de conexión con el servidor. ¿Está corriendo app.py?", "bot");
  }
});

// Manejo de subida de PDF
pdfInputEl.addEventListener("change", async () => {
  if (!pdfInputEl.files.length) return;

  const file = pdfInputEl.files[0];
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    addMessage("Solo se permiten archivos PDF.", "bot");
    pdfInputEl.value = "";
    return;
  }

  uploadStatusEl.textContent = `Subiendo ${file.name}...`;
  const formData = new FormData();
  formData.append("file", file);

  addMessage(`Subiendo PDF: ${file.name}`, "user");

  try {
    const res = await fetch("/upload_pdf", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      uploadStatusEl.textContent = "Error al subir PDF";
      addMessage("Error al subir PDF: " + (data.error || "desconocido"), "bot");
      pdfInputEl.value = "";
      return;
    }

    uploadStatusEl.textContent = `PDF indexado (${data.chunks} fragmentos).`;
    addMessage(
      `He indexado el PDF "${data.fileName}" con ${data.chunks} fragmentos.\n` +
        "Ahora puedes hacer preguntas sobre su contenido.",
      "bot"
    );
  } catch (err) {
    console.error(err);
    uploadStatusEl.textContent = "Error de conexión al subir PDF.";
    addMessage("Error de conexión al subir el PDF. Revisa si app.py está corriendo.", "bot");
  } finally {
    pdfInputEl.value = "";
  }
});
