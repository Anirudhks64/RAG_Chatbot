export default function GameFrame() {
  return (
    <div style={{ width: "100%", maxWidth: 640, aspectRatio: "16 / 9" }}>
      <iframe
        src={`/public/game.html?sid=${props.sid}`}
        title="Feather Fury"
        style={{ width: "100%", height: "100%", border: 0, borderRadius: 12 }}
      />
    </div>
  );
}