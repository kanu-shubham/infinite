import '@testing-library/jest-dom';

/**
 * jsdom doesn't ship IntersectionObserver. Hooks like useOnScreen and
 * useInfiniteScroll instantiate one, so we install a minimal mock that
 * records the observer instance per element. Tests can drive intersection
 * via the global __triggerIntersect helper attached to the constructor.
 */
class MockIntersectionObserver implements IntersectionObserver {
  readonly root: Element | Document | null = null;
  readonly rootMargin: string = '';
  readonly thresholds: ReadonlyArray<number> = [];

  private callback: IntersectionObserverCallback;
  private elements = new Set<Element>();

  constructor(cb: IntersectionObserverCallback, options?: IntersectionObserverInit) {
    this.callback = cb;
    if (options?.root && options.root !== null) {
      // Cast is fine — only used by tests, type is permissive on Document.
      this.root = options.root as Element;
    }
    this.rootMargin = options?.rootMargin ?? '0px';
    this.thresholds = Array.isArray(options?.threshold)
      ? (options!.threshold as number[])
      : [options?.threshold ?? 0];
    MockIntersectionObserver._instances.add(this);
  }

  observe(el: Element): void {
    this.elements.add(el);
  }
  unobserve(el: Element): void {
    this.elements.delete(el);
  }
  disconnect(): void {
    this.elements.clear();
    MockIntersectionObserver._instances.delete(this);
  }
  takeRecords(): IntersectionObserverEntry[] {
    return [];
  }

  static _instances = new Set<MockIntersectionObserver>();

  /** Fire intersection for every observed element across all live observers. */
  static triggerAll(isIntersecting: boolean): void {
    MockIntersectionObserver._instances.forEach((obs) => {
      const entries = Array.from(obs.elements).map((el) => ({
        isIntersecting,
        target: el,
        intersectionRatio: isIntersecting ? 1 : 0,
        boundingClientRect: el.getBoundingClientRect(),
        intersectionRect: el.getBoundingClientRect(),
        rootBounds: null,
        time: Date.now(),
      })) as unknown as IntersectionObserverEntry[];
      obs.callback(entries, obs as unknown as IntersectionObserver);
    });
  }
}

(global as unknown as { IntersectionObserver: typeof IntersectionObserver }).IntersectionObserver =
  MockIntersectionObserver as unknown as typeof IntersectionObserver;

// ResizeObserver — used by virtualizer hooks.
class MockResizeObserver implements ResizeObserver {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}
(global as unknown as { ResizeObserver: typeof ResizeObserver }).ResizeObserver =
  MockResizeObserver as unknown as typeof ResizeObserver;

export { MockIntersectionObserver };
