import React, { memo } from "react";
import "./Comment.css";

const Comment = memo(function Comment({ measureRef, comment }) {
  return (
    <li className="comment-item" ref={measureRef}>
      <span>
        [{comment.id}] {comment.email}
      </span>
      <p>{comment.body}</p>
    </li>
  );
});

export default Comment;
