export class EventTimeline {
  constructor(container = null) {
    this.container = container;
    this.events = [];
  }

  add(event) {
    const item = {
      event: event.event || event.type || "message",
      timestamp: event.timestamp || Date.now() / 1000,
      payload: event.payload || {},
    };
    this.events.push(item);
    this.renderItem(item);
    return item;
  }

  renderItem(item) {
    if (!this.container) return;
    const row = document.createElement("li");
    const time = new Date(item.timestamp * 1000).toLocaleTimeString();
    row.innerHTML = `<time>${time}</time><strong>${item.event}</strong><span>${this.summary(item.payload)}</span>`;
    this.container.prepend(row);
  }

  summary(payload) {
    if (typeof payload === "string") return payload;
    if (payload.text) return payload.text;
    if (payload.error) return payload.error;
    return JSON.stringify(payload);
  }
}
