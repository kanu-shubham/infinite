import React, { useEffect } from 'react';
import useOnScreen from '../hooks/useOnScreen';
import Comment from './Comment';
import type { CommentDto } from '../utils/api';
import './CommentList.css';

export interface CommentListProps {
  hasMore: boolean;
  isLoading: boolean;
  loadMore: () => void;
  comments: CommentDto[];
}

export default function CommentList({
  hasMore,
  isLoading,
  loadMore,
  comments,
}: CommentListProps): JSX.Element {
  const { measureRef, isIntersecting, observer } = useOnScreen();

  useEffect(() => {
    if (isIntersecting && hasMore) {
      loadMore();
      observer?.disconnect();
    }
    // loadMore intentionally omitted to match original behaviour.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isIntersecting, hasMore]);

  return (
    <ul className="comment-list">
      {comments.map((comment, index) => {
        if (index === comments.length - 1) {
          return (
            <Comment
              mesureRef={measureRef}
              key={comment.id}
              comment={comment}
            />
          );
        }
        return <Comment key={comment.id} comment={comment} />;
      })}
      {isLoading && <li>Loading...</li>}
    </ul>
  );
}
