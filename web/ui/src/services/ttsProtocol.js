export const PROTOCOL_VERSION = 1;

export function pingMessage() {
  return JSON.stringify({ protocol_version: PROTOCOL_VERSION, type: "ping" });
}

export function stopMessage() {
  return JSON.stringify({ protocol_version: PROTOCOL_VERSION, type: "stop" });
}

export function chatMessage({ text, history, nfeStep, speed, temperature, clientTurnId, screen }) {
  const message = {
    protocol_version: PROTOCOL_VERSION,
    type: "chat",
    text,
    history,
    nfe_step: nfeStep,
    client_turn_id: String(clientTurnId),
  };
  if (speed !== undefined) message.speed = speed;
  if (temperature !== undefined) message.temperature = temperature;
  if (screen) message.screen = screen;
  return JSON.stringify(message);
}
