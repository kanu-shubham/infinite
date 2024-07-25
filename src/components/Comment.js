import React from "react";
import "./Comment.css";

export default function Comment({ mesureRef, comment }) {
  return (
    <li className="comment-item" ref={mesureRef}>
      <span>
        [{comment.id}] {comment.email}
      </span>
      <p>{comment.body}</p>
    </li>
  );
}
