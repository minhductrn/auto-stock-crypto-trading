interface HeaderProps {
  symbols: string[];
  selected: string;
  onSelect: (symbol: string) => void;
}

export function Header({ symbols, selected, onSelect }: HeaderProps) {
  return (
    <header className="header">
      <div>
        <h1>AI-SDF Trading Console</h1>
        <span className="tag">Stage 1 · US Stocks · Daily · Long-only</span>
      </div>
      <label className="symbol-picker">
        Symbol
        <select value={selected} onChange={(e) => onSelect(e.target.value)}>
          {symbols.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </label>
    </header>
  );
}
