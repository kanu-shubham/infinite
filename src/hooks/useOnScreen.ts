import { useCallback, useState } from 'react';

export interface UseOnScreenOptions {
  root?: Element | null;
  rootMargin?: string;
  threshold?: number | number[];
}

export interface UseOnScreenResult {
  measureRef: (node: Element | null) => void;
  isIntersecting: boolean;
  observer: IntersectionObserver | undefined;
}

const useOnScreen = ({
  root = null,
  rootMargin = '0px',
  threshold = 0,
}: UseOnScreenOptions = {}): UseOnScreenResult => {
  const [observer, setObserver] = useState<IntersectionObserver | undefined>();
  const [isIntersecting, setIntersecting] = useState(false);

  const measureRef = useCallback(
    (node: Element | null) => {
      if (node) {
        const io = new IntersectionObserver(
          ([entry]) => setIntersecting(entry.isIntersecting),
          { root, rootMargin, threshold },
        );

        io.observe(node);
        setObserver(io);
      }
    },
    [root, rootMargin, threshold],
  );

  return { measureRef, isIntersecting, observer };
};

export default useOnScreen;
