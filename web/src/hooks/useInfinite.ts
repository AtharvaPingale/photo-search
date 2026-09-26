import { useCallback, useEffect, useRef, useState } from "react";

/** Paged loading driven by an IntersectionObserver sentinel below the grid.
 *  `key` identifies the query; changing it resets the list. */
export function useInfinite<T>(key: string, fetchPage: (offset: number) => Promise<T[]>, pageSize: number) {
  const [items, setItems] = useState<T[]>([]);
  const [loading, setLoading] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const gen = useRef(0);
  const busy = useRef(false);
  const fetchRef = useRef(fetchPage);
  fetchRef.current = fetchPage;
  const count = useRef(0);

  const load = useCallback(async () => {
    if (busy.current) return;
    busy.current = true;
    const g = gen.current;
    setLoading(true);
    try {
      const page = await fetchRef.current(count.current);
      if (g !== gen.current) return; // a newer query started meanwhile
      count.current += page.length;
      setItems((prev) => [...prev, ...page]);
      if (page.length < pageSize) setDone(true);
    } catch (e) {
      if (g === gen.current) {
        setError(e instanceof Error ? e.message : String(e));
        setDone(true);
      }
    } finally {
      if (g === gen.current) setLoading(false);
      busy.current = false;
    }
  }, [pageSize]);

  useEffect(() => {
    gen.current += 1;
    busy.current = false;
    count.current = 0;
    setItems([]);
    setDone(false);
    setError(null);
    void load();
  }, [key, load]);

  const sentinel = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = sentinel.current;
    if (!el || done) return;
    // start loading well before the user reaches the bottom
    const io = new IntersectionObserver((entries) => entries[0].isIntersecting && void load(), { rootMargin: "1200px" });
    io.observe(el);
    return () => io.disconnect();
  }, [load, done, items.length]);

  return { items, setItems, loading, done, error, sentinel };
}
