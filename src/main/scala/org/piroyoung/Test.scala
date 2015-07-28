package org.piroyoung

import org.piroyoung.classficate.FeedForwardNetwork
import org.piroyoung.linalg.ColVector

/**
 * Created by piroyoung on 7/25/15.
 */
object Test {
  def main(args: Array[String]) {

    val ff = FeedForwardNetwork(3, 8, 2, 2, 3)

    val input = ColVector(Seq(1,0,0))
    val answer = ColVector(Seq(1,0,0))

    val input2 = ColVector(Seq(1,0,1))
    val answer2 = ColVector(Seq(0,1,0))

    ff.fit(input, answer,iter=100)
    ff.fit(input2, answer2,iter=100)
    println("---")
    println("signal1:" + ff.predict(input).t.toString)
    println("signal2:" + ff.predict(input2).t.toString)


  }
}
