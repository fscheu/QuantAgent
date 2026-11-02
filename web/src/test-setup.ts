// jsdom no trae ResizeObserver; Recharts (ResponsiveContainer) lo necesita.
globalThis.ResizeObserver ??= class { observe() {} unobserve() {} disconnect() {} }
