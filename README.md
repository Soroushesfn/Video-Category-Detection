# The Project
The repository of the second phase of the main project of the Data Science course at the University of Tehran.

In this project, the US Trending YouTube videos dataset, a dataset containing various information about trending YouTube videos in the United States region, has been analyzed and has undergone preprocessing and feature engineering processes.

# The Database
At the initial step, the dataset has been imported into the SQL database format, where a single table with the following schema has been created:
* **Video ID:** The unique ID of each YouTube video
* **Title**: Video's title
* **Channel Title**: Name of the channel
* **Tags**: Tags included by the video
* **comments_disabled**: Whether the comments of the video have been disabled
* **ratings_disabled**: Whether the ratings of the video have been disabled
* **video_error_or_removed**: Whether the video contains a technical error
* **description**: Video's description
* **trending_date**: The date on which the video has been the trend
* **Views**: The number of views of the video
* **publish_date**: Publish date of the video
* **Publish_hour**: The hour at which the video was published
* **Category_id**: The category number of the video
* **likes**: Number of likes the video has
* **Dislikes**: Number of dislikes of the video
* **Comment_count**: Number of comments the video has

The mentioned schema forms our total dataset, in which there exist around 41000 records, containing the features above for each video.

# Feature Engineering
In this stage, the data is feature engineered via multiple steps. At first, a couple of meaningful columns, such as the engagement ratio column, have been added. In the second step, the data's textual features have been embedded through the help of a sentence transformer model, mapping each text sentence into a 384-dimensional vector. Three fields, the description, the title, and the tags of the video, have been embedded this way. This stage ends up with a couple of other minor actions on the data.

# Preprocessing
In this step, the engineered data yielded from the previous section has been preprocessed, and a good set of actions has been performed in order to keep the data as informative as possible. The data's rows with missing values have been omitted, the columns having a high correlation have been trimmed to a single specific column, etc., hoping to result in a more informative dataset that can help the ultimate mainstream task reach a better result.

# The Assets
The embeddings used in the feature engineering process have been stored separately in the **embeddings** folder, where there are a total of three sets of embeddings, corresponding to the textual attributes. The output of each stage has the form of Pandas **Dataframe**, stored in a Python **Pickle** file, where the initial data is fed to the feature engineering process, and its output is used as the input of the preprocess script, and finally, the output is treated as the final dataset.

# The Pipeline
The scripts are executed in the aforementioned format in the pipeline script, and the whole process is automated through the github's action tool, which runs the proper script upon each push. The screenshot of successful action runs is available in the submitted file.

# The Task
As of this stage, we have the task of **Video Category Classification** in mind and the operations performed in this phase have been aligned towards fulfilling this goal, hoping to obtain a dataset which is pointed more towards this task of ours, a task that yet has to be precisely defined and clarified.
