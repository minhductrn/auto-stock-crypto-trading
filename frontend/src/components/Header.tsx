interface HeaderProps {
  symbols: string[];
  selected: string;
  referenceSymbols?: string[];
  onSelect: (symbol: string) => void;
}

export function Header({ symbols, selected, onSelect, referenceSymbols = [] }: HeaderProps) {
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
              {s === '^GSPC' ? 'S&P 500 (^GSPC)' : s}{referenceSymbols.includes(s) ? ' · Chart only' : ''}
            </option>
          ))}
        </select>
      </label>
    </header>
  );
}
