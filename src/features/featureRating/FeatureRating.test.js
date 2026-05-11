import React from "react";
import { render, screen, fireEvent, act } from "@testing-library/react";
import "@testing-library/jest-dom";
import FeatureRating from "./FeatureRating";

function setup() {
  render(<FeatureRating />);
  return {
    triggerBtn: () => screen.getByRole("button", { name: /rate this feature/i }),
    thumbsDown: () => screen.getByRole("button", { name: /negative/i }),
    thumbsUp: () => screen.getByRole("button", { name: /positive/i }),
    stellar: () => screen.getByRole("button", { name: /stellar/i }),
  };
}

describe("FeatureRating", () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  test("renders trigger button", () => {
    const { triggerBtn } = setup();
    expect(triggerBtn()).toBeInTheDocument();
  });

  test("opens rating popup on trigger click", () => {
    const { triggerBtn } = setup();
    fireEvent.click(triggerBtn());
    expect(screen.getByText(/how would you rate this feature/i)).toBeInTheDocument();
  });

  test("closes popup when backdrop is clicked", () => {
    const { triggerBtn } = setup();
    fireEvent.click(triggerBtn());
    fireEvent.click(document.querySelector(".overlay__backdrop"));
    expect(screen.queryByText(/how would you rate this feature/i)).not.toBeInTheDocument();
  });

  describe("POSITIVE rating flow", () => {
    test("shows thank you popup after positive rating", () => {
      const { triggerBtn, thumbsUp } = setup();
      fireEvent.click(triggerBtn());
      fireEvent.click(thumbsUp());
      expect(screen.getByText(/thanks for your feedback/i)).toBeInTheDocument();
    });

    test("thank you popup auto-dismisses after 2 seconds", () => {
      const { triggerBtn, thumbsUp } = setup();
      fireEvent.click(triggerBtn());
      fireEvent.click(thumbsUp());
      expect(screen.getByText(/thanks for your feedback/i)).toBeInTheDocument();
      act(() => jest.advanceTimersByTime(2000));
      expect(screen.queryByText(/thanks for your feedback/i)).not.toBeInTheDocument();
    });

    test("does NOT show Trustpilot popup after positive rating", () => {
      const { triggerBtn, thumbsUp } = setup();
      fireEvent.click(triggerBtn());
      fireEvent.click(thumbsUp());
      act(() => jest.advanceTimersByTime(2000));
      expect(screen.queryByText(/trustpilot/i)).not.toBeInTheDocument();
    });
  });

  describe("NEGATIVE rating flow", () => {
    test("shows negative feedback form after thumbs down", () => {
      const { triggerBtn, thumbsDown } = setup();
      fireEvent.click(triggerBtn());
      fireEvent.click(thumbsDown());
      expect(screen.getByText(/how can we make things better/i)).toBeInTheDocument();
    });

    test("submit button is disabled when textarea is empty", () => {
      const { triggerBtn, thumbsDown } = setup();
      fireEvent.click(triggerBtn());
      fireEvent.click(thumbsDown());
      expect(screen.getByRole("button", { name: /submit/i })).toBeDisabled();
    });

    test("submit button is enabled after typing feedback", () => {
      const { triggerBtn, thumbsDown } = setup();
      fireEvent.click(triggerBtn());
      fireEvent.click(thumbsDown());
      fireEvent.change(screen.getByRole("textbox"), { target: { value: "needs improvement" } });
      expect(screen.getByRole("button", { name: /submit/i })).not.toBeDisabled();
    });

    test("shows thank you popup after submitting negative feedback", () => {
      const { triggerBtn, thumbsDown } = setup();
      fireEvent.click(triggerBtn());
      fireEvent.click(thumbsDown());
      fireEvent.change(screen.getByRole("textbox"), { target: { value: "needs improvement" } });
      fireEvent.click(screen.getByRole("button", { name: /submit/i }));
      expect(screen.getByText(/thanks for your feedback/i)).toBeInTheDocument();
    });

    test("does NOT show Trustpilot popup after negative feedback", () => {
      const { triggerBtn, thumbsDown } = setup();
      fireEvent.click(triggerBtn());
      fireEvent.click(thumbsDown());
      fireEvent.change(screen.getByRole("textbox"), { target: { value: "needs improvement" } });
      fireEvent.click(screen.getByRole("button", { name: /submit/i }));
      act(() => jest.advanceTimersByTime(2000));
      expect(screen.queryByText(/trustpilot/i)).not.toBeInTheDocument();
    });
  });

  describe("STELLAR rating flow", () => {
    test("shows thank you popup after stellar rating", () => {
      const { triggerBtn, stellar } = setup();
      fireEvent.click(triggerBtn());
      fireEvent.click(stellar());
      expect(screen.getByText(/thanks for your feedback/i)).toBeInTheDocument();
    });

    test("shows Trustpilot popup after thank you dismisses", () => {
      const { triggerBtn, stellar } = setup();
      fireEvent.click(triggerBtn());
      fireEvent.click(stellar());
      act(() => jest.advanceTimersByTime(2000));
      expect(screen.getByText(/enjoying bunq/i)).toBeInTheDocument();
    });

    test("closes Trustpilot popup on dismiss", () => {
      const { triggerBtn, stellar } = setup();
      fireEvent.click(triggerBtn());
      fireEvent.click(stellar());
      act(() => jest.advanceTimersByTime(2000));
      fireEvent.click(screen.getByRole("button", { name: /close/i }));
      expect(screen.queryByText(/enjoying bunq/i)).not.toBeInTheDocument();
    });

    test("Trustpilot CTA links to Trustpilot", () => {
      const { triggerBtn, stellar } = setup();
      fireEvent.click(triggerBtn());
      fireEvent.click(stellar());
      act(() => jest.advanceTimersByTime(2000));
      const link = screen.getByRole("link", { name: /go to trustpilot/i });
      expect(link).toHaveAttribute("href", expect.stringContaining("trustpilot.com"));
      expect(link).toHaveAttribute("target", "_blank");
    });
  });
});
