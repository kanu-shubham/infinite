import React, { Ref } from 'react';
import type { CommentDto } from '../utils/api';
import './Comment.css';

export interface CommentProps {
  comment: CommentDto;
  mesureRef?: Ref<HTMLLIElement>;
}

export default function Comment({ mesureRef, comment }: CommentProps): JSX.Element {
  return (
    <li className="comment-item" ref={mesureRef}>
      <span>
        [{comment.id}] {comment.email}
      </span>
      <p>{comment.body}</p>
    </li>
  );
}
