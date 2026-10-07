export type QuestionType = 'choice' | 'score' | 'noul'

export type ChoiceQuestion = {
  type: 'choice'
  instructions: string
  criteria: Record<string, string>
}

export type ScoreQuestion = {
  type: 'score'
  instructions: string
  criteria: string[]
}

export type NoulQuestion = {
  type: 'noul'
  instructions: string
}

export type Question = ChoiceQuestion | ScoreQuestion | NoulQuestion

export type SystemOneRequest = {
  model?: string
  state: string
  questions: Record<string, Question>
}

export type ChoiceAnswer = {
  type: 'choice'
  choice: string
  probabilities: Record<string, number>
  confidence?: number
}

export type ScoreAnswer = {
  type: 'score'
  score: number
  legend: Record<string, string>
  probabilities: Record<string, number>
  confidence?: number
}

export type NoulAnswer = {
  type: 'noul'
  noul: number
}

export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer

export type SystemOneResponse = {
  model: string
  answers: Record<string, Answer>
  usage?: { input_tokens: number; output_tokens: number }
  _mock?: boolean
}
