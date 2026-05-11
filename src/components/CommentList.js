import React, { useEffect } from "react";
import useOnScreen from "../hooks/useOnScreen";
import Comment from "./Comment";
import "./CommentList.css";

export default function CommentList({
  hasMore,
  isLoading,
  loadMore,
  comments
}) {
  const { measureRef, isIntersecting, disconnect } = useOnScreen();

  useEffect(() => {
    if (isIntersecting && hasMore) {
      loadMore();
      disconnect();
    }
  }, [isIntersecting, hasMore, loadMore, disconnect]);

  return (
    <ul className="comment-list">
      {comments.map((comment, index) => (
        <Comment
          key={comment.id}
          measureRef={index === comments.length - 1 ? measureRef : undefined}
          comment={comment}
        />
      ))}
      {isLoading && <li>Loading...</li>}
    </ul>
  );
}
