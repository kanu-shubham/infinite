// JSDoc typedefs for the memory library. Consumed by editors and
// `tsc --allowJs --checkJs` without requiring a TypeScript migration.

/**
 * @typedef {Map<string, number> | Float32Array | number[]} Vector
 */

/**
 * @typedef {Object} Embedder
 * @property {string} name
 * @property {number | null} dim - null for sparse bag-of-words vectors
 * @property {(text: string) => Promise<Vector>} encode
 * @property {(a: Vector, b: Vector) => number} sim
 * @property {(texts: string[]) => Promise<Vector[]>} [batch]
 */

/**
 * @typedef {Object} StorageAdapter
 * @property {() => Promise<void>} open
 * @property {(key: string) => Promise<any | null>} get
 * @property {(key: string, value: any) => Promise<void>} set
 * @property {(key: string) => Promise<void>} del
 * @property {(prefix?: string) => Promise<string[]>} list
 * @property {() => Promise<number>} version
 * @property {(n: number) => Promise<void>} setVersion
 * @property {() => Promise<void>} close
 */

/**
 * @typedef {Object} Migration
 * @property {number} from
 * @property {number} to
 * @property {(adapter: StorageAdapter) => Promise<void>} up
 */

/**
 * @typedef {Object} ShortTermEntry
 * @property {string} id
 * @property {string} text
 * @property {string} role
 * @property {number} salience
 * @property {number} tokens
 * @property {Vector} features
 * @property {number} createdAt
 * @property {Record<string, any>} meta
 */

/**
 * @typedef {Object} Episode
 * @property {string} id
 * @property {string} text
 * @property {string[]} tags
 * @property {Vector} features
 * @property {number} strength
 * @property {number} createdAt
 * @property {number} lastAccessed
 * @property {number} accessCount
 * @property {Record<string, any>} meta
 */

/**
 * @typedef {Object} GraphNode
 * @property {string} id
 * @property {string} type
 * @property {string} label
 * @property {Record<string, any>} props
 * @property {number} weight
 * @property {number} createdAt
 * @property {number} updatedAt
 */

/**
 * @typedef {Object} GraphEdge
 * @property {string} id
 * @property {string} from
 * @property {string} to
 * @property {string} relation
 * @property {number} weight
 * @property {Record<string, any>} props
 * @property {number} createdAt
 * @property {number} updatedAt
 */

/**
 * @typedef {Object} Signal
 * @property {"stm" | "ltm" | "kg" | "recency"} source
 * @property {string} id
 * @property {number} score - normalized to [0, 1]
 * @property {any} [payload]
 */

/**
 * @typedef {Object} RetrievalResult
 * @property {string} id
 * @property {number} score
 * @property {Signal[]} signals
 * @property {any} payload
 */

/**
 * @typedef {Object} MemoryEvent
 * @property {"stm:add" | "stm:evict" | "ltm:remember" | "ltm:recall"
 *   | "ltm:forget" | "kg:upsert" | "retrieval:done" | "feedback:applied"
 *   | "decay:run" | "storage:quotaExceeded" | "manager:ready"
 *   | "manager:disposed"} type
 * @property {number} at
 * @property {any} [data]
 */

export {};
