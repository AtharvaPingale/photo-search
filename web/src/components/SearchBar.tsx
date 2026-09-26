import { useEffect, useRef, useState } from "react";

import { useSpeech } from "../hooks/useSpeech";
import { Icon } from "./Icon";

type Props = {
  value: string;
  onSearch: (q: string) => void;
  onImage: (file: File) => void;
  busy?: boolean;
};

export function SearchBar({ value, onSearch, onImage, busy }: Props) {
  const [text, setText] = useState(value);
  const input = useRef<HTMLInputElement>(null);
  const file = useRef<HTMLInputElement>(null);
  const speech = useSpeech((t) => {
    setText(t);
    onSearch(t);
  });

  useEffect(() => setText(value), [value]);

  return (
    <form
      className="searchbar"
      onSubmit={(e) => {
        e.preventDefault();
        input.current?.blur(); // dismiss the phone keyboard so results are visible
        onSearch(text.trim());
      }}
    >
      <span className="search-icon" aria-hidden>
        {busy ? <span className="spinner tiny" /> : <Icon name="search" size={18} />}
      </span>
      <input
        ref={input}
        type="search"
        enterKeyHint="search"
        autoCapitalize="off"
        autoCorrect="off"
        placeholder={speech.listening ? "Listening…" : "golden hour on the beach"}
        value={speech.listening && speech.interim ? speech.interim : text}
        onChange={(e) => setText(e.target.value)}
      />
      {text && !speech.listening && (
        <button type="button" className="icon" aria-label="clear" onClick={() => { setText(""); onSearch(""); }}>
          <Icon name="x" size={18} />
        </button>
      )}
      {speech.supported && (
        <button
          type="button"
          className={`icon${speech.listening ? " recording" : ""}`}
          aria-label={speech.listening ? "stop voice search" : "voice search"}
          onClick={() => (speech.listening ? speech.stop() : speech.start())}
        >
          <Icon name="mic" />
        </button>
      )}
      <button type="button" className="icon" aria-label="search by photo" onClick={() => file.current?.click()}>
        <Icon name="image" />
      </button>
      <input
        ref={file}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onImage(f);
          e.target.value = "";
        }}
      />
      {speech.error && <div className="searchbar-error">{speech.error}</div>}
    </form>
  );
}
