package com.example.ollamacameraapp.network

import android.util.Base64
import android.util.Log
import com.example.ollamacameraapp.model.Content
import com.example.ollamacameraapp.model.ImageUrl
import com.example.ollamacameraapp.model.Instance
import com.example.ollamacameraapp.model.Message
import com.example.ollamacameraapp.model.VertexRequest
import com.example.ollamacameraapp.model.VertexResponse
import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.json.Json
import java.io.File

private val json = Json { ignoreUnknownKeys = true }

class VertexAiClient {
    private val client = HttpClient(CIO) {
        install(ContentNegotiation) {
            json(Json {
                prettyPrint = true
                isLenient = true
                ignoreUnknownKeys = true
            })
        }
    }

    suspend fun getDescription(file: File, accessToken: String): String {
        val rawResponse: String = try {
            client.post("https://europe-west4-aiplatform.googleapis.com/v1/projects/gemma-hcls25par-723/locations/europe-west4/endpoints/1291110325008990208:predict") {
                header("Authorization", "Bearer $accessToken")
                contentType(ContentType.Application.Json)
                setBody(VertexRequest(
                    instances = listOf(
                        Instance(
                            requestFormat = "chatCompletions",
                            messages = listOf(
                                Message(
                                    role = "user",
                                    content = listOf(
                                        Content(
                                            type = "text",
                                            text = "Describe the picture in one sentence. example: A person holding a smartphone."
                                        ),
                                        Content(
                                            type = "image_url",
                                            image_url = ImageUrl(url = "data:image/jpeg;base64," + Base64.encodeToString(file.readBytes(), Base64.NO_WRAP))
                                        )
                                    )
                                )
                            )
                        )
                    )
                ))
            }.bodyAsText()
        } catch (e: Exception) {
            Log.e("VertexAiClient", "Error making request", e)
            return "Error: ${e.message}"
        }

        Log.d("VertexResponse", rawResponse)
        val response = json.decodeFromString<VertexResponse>(rawResponse)
        return response.getContent() ?: "No description found"
    }
}
