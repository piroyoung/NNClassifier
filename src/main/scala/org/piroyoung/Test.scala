package org.piroyoung

import org.piroyoung.classficate.FeedForwardNetwork
import org.piroyoung.linalg.RowVector

/**
 * Created by piroyoung on 7/25/15.
 */
object Test {
  def main(args: Array[String]) {

    val ff = new FeedForwardNetwork(3, 2, 4, 5, 2)

    ff.forward(RowVector(Seq(1,1,5))).foreach(x => println(x.toString + "\n---"))

//    val in  = ff.layers(0).getActivated(RowVector(Seq(1,1,1)).addBias)
//    println(ff.layers(0).weights)
//    println(in.toString)

//    val v = new RowVector(Seq(2,2,2))
//
//    println(v.toString)
//    println(v.addBias.toString)

    println((RowVector(Seq(1,2,3)) * RowVector(Seq(3,2,1))).toString)

  }
}
