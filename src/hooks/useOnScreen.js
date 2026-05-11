import { useCallback, useState, useRef, useEffect } from "react";

const useOnScreen = ({
  root = null,
  rootMargin = "0px",
  threshold = 0
} = {}) => {
  const observerRef = useRef(null);
  const [isIntersecting, setIntersecting] = useState(false);

  // Disconnect on unmount to prevent memory leak.
  useEffect(() => () => observerRef.current?.disconnect(), []);

  const measureRef = useCallback(
    (node) => {
      // Disconnect previous observer before creating a new one.
      observerRef.current?.disconnect();
      observerRef.current = null;
      if (!node) return;
      observerRef.current = new IntersectionObserver(
        ([entry]) => setIntersecting(entry.isIntersecting),
        { root, rootMargin, threshold }
      );
      observerRef.current.observe(node);
    },
    [root, rootMargin, threshold]
  );

  const disconnect = useCallback(() => {
    observerRef.current?.disconnect();
    observerRef.current = null;
  }, []);

  return { measureRef, isIntersecting, disconnect };
};

export default useOnScreen;
