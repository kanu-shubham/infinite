import type { AxiosInstance, AxiosResponse } from 'axios';
import { API } from './constants';

export interface CommentDto {
  postId: number;
  id: number;
  name: string;
  email: string;
  body: string;
}

export default class Api {
  private client: AxiosInstance;

  constructor(httpClient: AxiosInstance) {
    this.client = httpClient;
  }

  async comments(page: number): Promise<AxiosResponse<CommentDto[]>> {
    return this.client.get<CommentDto[]>('comments', {
      params: { _page: page, _limit: API.limit },
    });
  }
}
